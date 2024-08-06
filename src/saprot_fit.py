from typing import Tuple
from pathlib import Path
from functools import partial


import numpy as np
import pandas as pd
import ast

import tqdm
import h5py
import hydra
from omegaconf import DictConfig
 
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score

from scipy.stats import spearmanr

from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint




#weird error
#from utils.downstream_utils import count_f1_max, save_results_to_csv


import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


#for Bitvectors
def identity_map(embedding, cfg):
    return embedding

def threshold_map(embedding, cfg):
    dev = embedding.device
    return torch.where(embedding > float(cfg.transform_func.split("_")[1]), torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev))

# Define the new dense map
def dense_value_map(embedding, cfg):
    # Normalizing the embedding to be between 0 and 1
    min_val = torch.min(embedding)
    max_val = torch.max(embedding)
    dense_embedding = (embedding - min_val) / (max_val - min_val)
    return dense_embedding

def dense_pca_map(embedding, cfg):
    n_components = int(cfg.transform_func.split("_")[1])  # Number of dimensions to reduce to
    pca = PCA(n_components=n_components)
    
    # Convert the embedding to a numpy array, apply PCA, and convert back to a tensor
    embedding_np = embedding.cpu().detach().numpy()
    dense_embedding_np = pca.fit_transform(embedding_np)
    dense_embedding = torch.tensor(dense_embedding_np).to(embedding.device)
    
    return dense_embedding

function_map = {
    "identity": identity_map,
    "threshold": threshold_map,
    "densevalue": dense_value_map,
    "densepca": dense_pca_map,
}


#Search Grid for Randon Forest classifier 
# Hyperparameter grid
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    #'max_depth': [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


def count_f1_max(pred, target): 
	"""
	    F1 score with the optimal threshold, Copied from TorchDrug.

	    This function first enumerates all possible thresholds for deciding positive and negative
	    samples, and then pick the threshold with the maximal F1 score.

	    Parameters:
	        pred (Tensor): predictions of shape :math:`(B, N)`
	        target (Tensor): binary targets of shape :math:`(B, N)` 
    """
    
	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)

	
	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
	                torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
	             torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1.max()



def save_results_to_csv(results: dict, cfg: DictConfig) -> None:
    output_dir = Path(cfg.results_dir) / "downstream_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = "OneProt" if "oneprot" in cfg.model.network.modules.sequence.model_name_or_path else "ESM2"
    csv_path = output_dir / f"{cfg.task_name}.csv"
    
    results_df = pd.DataFrame(results)
    results_df["model_type"] = model_type
    results_df["classifier"] = str(cfg.classifier._target_)
    results_df["hyperparameters"] = str(cfg.classifier)
    
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")



def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    print(cfg)
    #output_dir = Path(cfg.output_dir) / cfg.task_name
    if cfg.system == "OneProt":
        output_dir = Path(cfg.oneprot_output_dir) / cfg.task_name
    elif cfg.system == "ESM2":
        output_dir = Path(cfg.esm2_output_dir) / cfg.task_name
    else: 
        print('Model not configured')
        return 
    
    # Create the transformation function
    func = partial(function_map[cfg.transform_func.split("_")[0]], cfg=cfg)


    all_inputs = dict()

    for partition in cfg.evaluate_on:
        partition_dir = output_dir / partition
        seq_embs_list = []
        mod_embs_list = []
        targets_list = []
        
        # Iterate through subfolders in the partition directory
        for k, subfolder in tqdm.tqdm(enumerate(partition_dir.iterdir())):
            if subfolder.is_dir():
                file_range = subfolder.glob("*.npy")
                file_range = [int(f.stem.split("_")[0]) for f in file_range]
                file_range = sorted(list(set(file_range)))
                for idx in file_range:
                    seq_embs_list.append(np.load(subfolder / f"{idx}_seq.npy"))
                    if cfg.fit_classifier_on in ["struct", "both"]:
                        mod_embs_list.append(np.load(subfolder / f"{idx}_mod.npy"))
                    targets_list.append(np.load(subfolder / f"{idx}_target.npy"))

        # Concatenate embeddings if there are multiple files
        seq_embs = np.concatenate(seq_embs_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        if cfg.fit_classifier_on in ["struct", "both"]:
            mod_embs = np.concatenate(mod_embs_list, axis=0)
            print(partition, seq_embs.shape, mod_embs.shape, targets.shape)
        else:
            print(partition, seq_embs.shape, targets.shape)

        

        if cfg.fit_classifier_on == "seq":
            embeddings = seq_embs
        elif cfg.fit_classifier_on == "struct":
            embeddings = mod_embs
        elif cfg.fit_classifier_on == "both":
            embeddings = np.concatenate([seq_embs, mod_embs], axis=1)
        else:
            raise NotImplementedError
        
        # Convert embeddings to torch.Tensor for transformation
        embeddings = torch.tensor(embeddings)
        # Apply the transformation function to the embeddings
        embeddings = func(embeddings)
        # Convert the transformed embeddings back to numpy array
        embeddings = embeddings.numpy()


        # Convert targets to numeric values if they are in string format
        if isinstance(targets[0], str):
            #if targets[0].startswith("["):  # Check if it is a list-like string
            #    targets = np.array([np.argmax(ast.literal_eval(t)) for t in targets], dtype=np.float64)
            #else:
            #    targets = np.array([ast.literal_eval(t) for t in targets], dtype=np.float64)
            targets = np.array([ast.literal_eval(t) for t in targets], dtype=np.float64)
        else:
            targets = np.array(targets, dtype=np.float64)

        # Ensure targets are one-dimensional or retain their original shape if they are one-hot encoded
        if targets.ndim == 1:
            targets = targets.squeeze()
        else:
            targets = targets
        
        

        # Convert targets to integer labels
        targets = targets.astype(int)



        print(f"Shape of embeddings: {embeddings.shape}")
        print(f"Shape of targets: {targets.shape}")

        # Create a mask to filter out rows with NaN values in embeddings and targets
        nan_mask_embeddings = ~np.isnan(embeddings).any(axis=1)
        nan_mask_targets = ~np.isnan(targets)

        #print(f"Shape of nan_mask_embeddings: {nan_mask_embeddings.shape}")
        #print(f"Shape of nan_mask_targets: {nan_mask_targets.shape}")

        #nan_mask = nan_mask_embeddings & nan_mask_targets

        # Apply the mask to filter out rows with NaN values
        #embeddings = embeddings[nan_mask]
        #targets = targets[nan_mask]

        #print(nan_mask.sum())

        all_inputs[f"{partition}_emb"] = embeddings
        all_inputs[f"{partition}_target"] = targets

    task_type = None

    if cfg.task_name in ["MetalIonBinding", "GO_MF", "GO_CC", "GO_BP", "EC", "DeepLoc_cls2", "DeepLoc_cls10", "HumanPPI"]:
        task_type = 'classification'

        if cfg.task_name in ["GO_BP", "GO_MF", "GO_CC", "EC"]:
            #classifier = hydra.utils.instantiate(cfg.classifier)
            #classifier.fit(all_inputs["train_emb"], all_inputs["train_target"])

            #TODO: DOING MULTI-LABEL Classification with Scikit-Learn does not work yet. Maybe check datasets or code
            base_classifier = hydra.utils.instantiate(cfg.classifier)
            classifier = MultiOutputClassifier(base_classifier)
            classifier.fit(all_inputs["train_emb"], all_inputs["train_target"])

        else: 
            classifier = hydra.utils.instantiate(cfg.classifier)

            if cfg.hyperparameter_search:
                print('Start RF Hyperparamteter search')
                rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, 
                                           n_iter = 100, 
                                           cv = 3, verbose=2, random_state=42, n_jobs = -1)
                # Fit the random search model
                rf_random.fit(all_inputs["train_emb"], all_inputs["train_target"])
                print('best parameters ', rf_random.best_params_)

                classifier = rf_random.best_estimator_

            else:
                classifier.fit(all_inputs["train_emb"], all_inputs["train_target"])

    elif cfg.task_name in ["Thermostability"]:
        task_type = 'regression'
        regressor = hydra.utils.instantiate(cfg.regressor)
        regressor.fit(all_inputs["train_emb"], all_inputs["train_target"])
    else:
        print("Task name not configured")
        raise Exception

    results = {}
    # train the classifier
    if task_type == 'classification':

        

        for partition in ["val", "test"]:
            print(f"Inference on {partition}...")
            

            #Does not work
            if cfg.task_name in ["GO_BP", "GO_MF", "GO_CC", "EC"]:
                y_pred_proba = classifier.predict_proba(all_inputs[f"{partition}_emb"]) 

                label2num = {"EC": 585, "GO_BP": 1943, "GO_MF": 489, "GO_CC": 320}
                num_classes = label2num[cfg.task_name]
                #self.num_labels = label2num[anno_type]
  

                y_pred_tensor = torch.tensor(y_pred_proba)
                print("y_pred_tensor shape ", y_pred_tensor.shape)
                
                target_tensor = torch.tensor(all_inputs[f"{partition}_target"])  # Add batch dimension
                # Find unique elements and their count
                unique_elements = torch.unique(target_tensor)
                num_unique_elements = unique_elements.size(0)
                max_element = torch.max(target_tensor)
                print(f"Number of unique elements: {num_unique_elements}")
                print(f"Maximum element: {max_element.item()}")

                print("y_pred_tensor ", y_pred_tensor[:5])
                print("target_tensor ", target_tensor[:5])

                #target_tensor_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()

                # Ensure correct reshaping for count_f1_max
                y_pred_reshaped = y_pred_tensor.view(-1, num_classes)
                target_reshaped = target_tensor.view(-1, num_classes)

                print("y_pred_tensor shape ", y_pred_reshaped.shape)
                print("target_tensor shape ", target_reshaped.shape)
                print("y_pred_tensor ", y_pred_reshaped[:5])
                print("target_tensor ", target_reshaped[:5])
                #score = count_f1_max(torch.tensor(y_pred).view(1,-1), torch.tensor(all_inputs[f"{partition}_target"]).view(1,-1))
                score = count_f1_max(y_pred_reshaped, target_tensor)
                print(f"{partition} Fmax Score: {score.item():.4f}")
                results[f"{partition}_fmax"] = score.item()

            else:
                

                y_pred = classifier.predict(all_inputs[f"{partition}_emb"])
                # Evaluate classifier
                accuracy = accuracy_score(all_inputs[f"{partition}_target"], y_pred)
                f1_micro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='micro')
                f1_macro = f1_score(all_inputs[f"{partition}_target"], y_pred, average='macro')
                conf_matrix = confusion_matrix(all_inputs[f"{partition}_target"], y_pred)

                print(f"{partition} Accuracy: {accuracy:.4f}")
                print(f"{partition} F1 Score (Micro): {f1_micro:.4f}")
                print(f"{partition} F1 Score (Macro): {f1_macro:.4f}")
                print(f"{partition} Confusion Matrix:\n{conf_matrix}")


                #misclassified_idx = [i for i, (true, pred) in enumerate(zip(all_inputs[f"{partition}_target"], y_pred)) if true != pred]
                #print(f"{partition} Misclassified indices: {misclassified_idx}")

                results[f"{partition}_accuracy"] = accuracy
                results[f"{partition}_f1_micro"] = f1_micro
                results[f"{partition}_f1_macro"] = f1_macro
                results[f"{partition}_conf_matrix"] = conf_matrix

   

    else: 
        for partition in ["val", "test"]:
            print(f"Inference on {partition}...")
            y_pred = regressor.predict(all_inputs[f"{partition}_emb"])

            mse = mean_squared_error(all_inputs[f"{partition}_target"], y_pred)
            r2 = r2_score(all_inputs[f"{partition}_target"], y_pred)
            spearman_rho, _ = spearmanr(all_inputs[f"{partition}_target"], y_pred)

            print(f"{partition} Mean Squared Error: {mse:.4f}")
            print(f"{partition} R2 Score: {r2:.4f}")
            print(f"{partition} Spearman's Rank Correlation Coefficient: {spearman_rho:.4f}")

            results[f"{partition}_mse"] = mse
            results[f"{partition}_r2"] = r2
            results[f"{partition}_spearman_rho"] = spearman_rho

    save_results_to_csv(results, cfg)



@hydra.main(version_base="1.3", config_path="../configs", config_name="saprot.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print('taskname: ', cfg.task_name)
    print('classifier: ', cfg.classifier)
    print('transform_func: ', cfg.transform_func)
    print('system: ', cfg.system)
    evaluate(cfg)




if __name__ == "__main__":
    main()
 