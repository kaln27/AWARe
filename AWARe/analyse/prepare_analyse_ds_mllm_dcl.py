import json
import random
import os
import argparse
from typing import Union
import argparse

SEED = 42

questions_path_map = {
    "RS": "datasets/RS/train.json",
    "Med": "datasets/Med/train.json",
    "AD": "datasets/AD/train.json",
    "Sci": "datasets/Sci/train.json",
    "Fin": "datasets/Fin/train.json",
}

image_dir_map = {
    "RS": "datasets/RS",
    "Med": "datasets/Med",
    "AD": "datasets/AD",
    "Sci": "datasets/Sci",
    "Fin": "datasets/Fin",
}

def prepare_ds(ds_name, each_ds_num, output_dir, seed=SEED):
    print(f"Preparing dataset: {ds_name} for AWARe training ...")
    output_path = os.path.join(output_dir, f"{ds_name}.jsonl")
    if os.path.exists(output_path):
        print(f"{ds_name} dataset already prepared. Skipping.")
        return
    questions_path = questions_path_map[ds_name]
    image_dir = image_dir_map[ds_name]

    with open(questions_path, 'r') as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)
    selected_data = data[:each_ds_num]

    formated_data = []
    for item in selected_data:
        question_id = item.get("question_id", "")
        image_path = os.path.join(image_dir, item["image"])
        conv = item["conversations"]
        text = conv[0]['value'].replace('<image>\n', '').strip()
        answer = conv[1]['value']
        formated_data.append({
            "question_id": question_id,
            "image": image_path,
            "text": text,
            "answer": answer
        })
    
    with open(output_path, 'w') as f:
        for item in formated_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(formated_data)} samples for {ds_name} to {output_dir}/{ds_name}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare analysis dataset for AWARe training.")
    parser.add_argument('--output_dir', type=str, default="outputs/analyse_ds",
                        help='Output directory for the prepared dataset')
    parser.add_argument('--each_ds_num', type=str, default="100", 
                        help='Number of samples to select from each dataset, split by comma or int')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name in questions_path_map.keys():
        prepare_ds(ds_name, int(args.each_ds_num), args.output_dir)