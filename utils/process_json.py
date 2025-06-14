import os
import json
import csv


def process_json_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)

            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            dataset = data.get("dataset", "")
            model = data.get("model", "")
            best_params = data.get("hyperparameter_search", {}).get("best_params", {})
            # If best_params is empty, use "grid_search" instead of "hyperparameter_search"
            if not best_params:
                best_params = data.get("grid_search", {}).get("best_params", {})
            accuracy = data.get("final_test_results", {}).get("metrics", {}).get("accuracy", "")
            f1_score = data.get("final_test_results", {}).get("metrics", {}).get("f1_score", "")
            training_time = data.get("final_test_results", {}).get("training_time", "")

            csv_file_name = f"{model}.csv"
            csv_file_path = os.path.join(output_dir, csv_file_name)

            # Check if the CSV file exists and read existing rows
            file_exists = os.path.isfile(csv_file_path)
            existing_rows = set()
            if file_exists:
                with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        existing_rows.add((row["dataset"], row["model"]))

            # Add a new row only if it does not exist
            if (dataset, model) not in existing_rows:
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)

                    # Add a header if the file does not exist
                    if not file_exists:
                        writer.writerow([
                            "dataset", "model", "best_params",
                            "final_test_results-accuracy",
                            "final_test_results-f1_score",
                            "final_test_results-training_time"
                        ])

                    writer.writerow([
                        dataset, model, json.dumps(best_params),
                        accuracy, f1_score, training_time
                    ])
