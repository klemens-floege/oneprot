import os
import csv
import json
import lmdb
import argparse
from tqdm import tqdm


def convert_lmdb_to_csv(lmdb_path, csv_path, dataset_type):
    env = lmdb.open(lmdb_path, lock=False, readonly=True)

    with env.begin() as txn:
        length = int(txn.get("length".encode()).decode())

        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Determine headers based on dataset type
            if dataset_type == "classification":
                headers = ["sequence", "label", "plddt", "coords"]
            elif dataset_type == "regression":
                headers = ["sequence", "fitness", "plddt"]
            else:
                headers = ["sequence", "label/fitness", "plddt", "coords"]

            csvwriter.writerow(headers)

            for i in tqdm(
                range(length), desc=f"Converting {os.path.basename(lmdb_path)}"
            ):
                entry = json.loads(txn.get(str(i).encode()).decode())
                sequence = entry["seq"]

                if dataset_type == "classification":
                    label = entry.get("label", "N/A")
                    plddt = json.dumps(entry.get("plddt", []))
                    coords = json.dumps(entry.get("coords", {}))
                    csvwriter.writerow([sequence, label, plddt, coords])
                elif dataset_type == "regression":
                    fitness = entry.get("fitness", "N/A")
                    plddt = json.dumps(entry.get("plddt", []))
                    csvwriter.writerow([sequence, fitness, plddt])
                else:
                    label_or_fitness = entry.get("label", entry.get("fitness", "N/A"))
                    plddt = json.dumps(entry.get("plddt", []))
                    coords = json.dumps(entry.get("coords", {}))
                    csvwriter.writerow([sequence, label_or_fitness, plddt, coords])


def process_folder(input_folder, output_folder, dataset_type):
    for root, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            lmdb_path = os.path.join(root, dir_name)
            if os.path.isfile(os.path.join(lmdb_path, "data.mdb")):
                # Extract the task name (first subfolder after input_folder)
                rel_path = os.path.relpath(root, input_folder)
                path_parts = rel_path.split(os.path.sep)
                task_name = path_parts[0] if path_parts[0] != "." else ""

                # Construct the remaining subfolder structure
                subfolder_structure = (
                    "_".join(path_parts[1:]) if len(path_parts) > 1 else ""
                )

                # Create a flattened filename
                if subfolder_structure:
                    csv_name = f"{task_name}_{subfolder_structure}_{dir_name}.csv"
                else:
                    csv_name = f"{task_name}_{dir_name}.csv"

                # Ensure the output folder exists
                os.makedirs(output_folder, exist_ok=True)

                csv_path = os.path.join(output_folder, csv_name)
                try:
                    convert_lmdb_to_csv(lmdb_path, csv_path, dataset_type)
                    print(f"Converted {lmdb_path} to {csv_path}")
                except:  # noqa: E722
                    print(f"COULD NOT CONVERT {lmdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LMDB files to CSV")
    parser.add_argument("input_folder", help="Path to the folder containing LMDB files")
    parser.add_argument(
        "output_folder", help="Path to the folder where CSV files will be saved"
    )
    parser.add_argument(
        "--type",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Type of dataset (classification, regression, or auto)",
    )
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.type)
    print(f"Conversion complete. CSV files are saved in {args.output_folder}")
