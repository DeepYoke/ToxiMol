import json
import os
import argparse

def extract_results(response_text):
    results = []
    if "MODIFIED_SMILES:" in response_text:
        smiles_part = response_text.split("MODIFIED_SMILES:")[1].strip()
        if ";" in smiles_part:
            smiles_candidates = [s.strip() for s in smiles_part.split(";") if s.strip()]
            results.extend(smiles_candidates[:3])
        else:
            if smiles_part.lower().strip() == "none":
                results = []
            else:
                results.append(smiles_part.strip())
    return results

def process_directory(root_path):
    for sub_dir in os.listdir(root_path):
        sub_dir_path = os.path.join(root_path, sub_dir)
        for file in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file)
            with open(file_path, 'r') as f:
                json_file = json.load(f)
                results = json_file["results"]
                for i in range(len(results)):
                    raw_response = results[i]["raw_response"][0]
                    new_response = extract_results(raw_response)
                    json_file["results"][i]["modified_smiles"] = new_response
            with open(file_path, 'w') as f:
                json.dump(json_file, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str)
    args = parser.parse_args()
    process_directory(args.root_path)
