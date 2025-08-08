import os
import csv
import json
import lmdb
import argparse
from tqdm import tqdm


def convert_lmdb_to_csv(lmdb_path, csv_path, dataset_type, plddt_threshold=None):
    env = lmdb.open(lmdb_path, lock=False, readonly=True)
    
    with env.begin() as txn:
        length = int(txn.get("length".encode()).decode())

        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Determine headers based on dataset type
            if dataset_type == "classification":
                headers = ["sequence", "label/fitness"]
            elif dataset_type == "regression":
                headers = ["sequence", "label/fitness" ]
            elif dataset_type == "ppi":
                headers = ["sequence_1", "sequence_2", "label/fitness"]
            else:
                headers = ["sequence", "label/fitness"]

            csvwriter.writerow(headers)

            for i in tqdm(
                range(length), desc=f"Converting {os.path.basename(lmdb_path)}"
            ):
                entry = json.loads(txn.get(str(i).encode()).decode())

                if dataset_type == "classification":
                    sequence = entry["seq"]
                    label = entry.get("label", "N/A")
                    csvwriter.writerow([sequence, label])
                elif dataset_type == "regression":
                    sequence = entry["seq"]
                    fitness = entry.get("fitness", "N/A")
                    csvwriter.writerow([sequence, fitness])
                elif dataset_type == "ppi":
                    seq_1, seq_2 = entry['seq_1'], entry['seq_2']
                    label = entry.get("label", "N/A")
                    csvwriter.writerow([seq_1, seq_2, label])
                else:
                    sequence = entry["seq"]
                    label_or_fitness = entry.get("label", entry.get("fitness", "N/A"))
                    csvwriter.writerow([sequence, label_or_fitness])


def process_folder(input_folder, output_folder, dataset_type, plddt_threshold=None):
     for root, dirs, files in os.walk(input_folder):
       
        if 'data.mdb' in files:
            lmdb_path = root
            # Extract the task name and subfolder structure
            rel_path = os.path.relpath(root, input_folder)
            path_parts = rel_path.split(os.path.sep)
            
            # Create a flattened filename
            csv_name = f"{path_parts[0]}_{path_parts[-1]}.csv"

            # Ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)

            csv_path = os.path.join(output_folder, csv_name)
            try:
                convert_lmdb_to_csv(lmdb_path, csv_path, dataset_type, plddt_threshold)
                print(f"Converted {lmdb_path} to {csv_path}")
            except Exception as e:
                print(f"COULD NOT CONVERT {lmdb_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LMDB files to CSV")
    parser.add_argument("--input_folder", help="Path to the folder containing LMDB files")
    parser.add_argument(
        "--output_folder", help="Path to the folder where CSV files will be saved"
    )
    parser.add_argument(
        "--type",
        choices=["classification", "regression", "ppi", "auto"],
        default="ppi",
        help="Type of dataset (classification, regression, ppi, or auto)",
    )
    parser.add_argument(
        "--plddt_threshold",
        type=float,
        default=None,
        help="pLDDT threshold for masking structure tokens (only applicable for PPI)",
    )
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.type, args.plddt_threshold)
    print(f"Conversion complete. CSV files are saved in {args.output_folder}")