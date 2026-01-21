import json
import random
import os
import argparse
from typing import Union

EACH_DS_NUM = 200
SEED = 42

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

questions_path_map = {
    "okvqa": "instructions/OKVQA/okvqa_val.jsonl",
    "ocrvqa": "instructions/OCRVQA/sampled_ocrvqa_test.jsonl",
    "gqa": "instructions/GQA/llava_gqa_testdev_balanced.jsonl",
    "textvqa": "instructions/TextVQA/llava_textvqa_val_v051_ocr.jsonl",
}


image_dir_map = {
    "okvqa": "datasets/COCO/val2014",
    "ocrvqa": "datasets/OCR-VQA/images",
    "gqa": "datasets/GQA/images",
    "textvqa": "datasets/TextVQA/train_images",
}

def prepare_ds(ds_name, each_ds_num=EACH_DS_NUM, seed=SEED):
    print(f"Preparing dataset: {ds_name} for AWARe training ...")
    questions_path = questions_path_map[ds_name]
    image_dir = image_dir_map[ds_name]

    data = load_jsonl(questions_path)
    random.seed(seed)
    random.shuffle(data)
    selected_data = data[:each_ds_num]

    for item in selected_data:
        item['image'] = os.path.join(image_dir, item['image'])
    
    return selected_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_ds', type=str, default="okvqa,ocrvqa,gqa,textvqa", 
                        help='Target dataset to prepare (default: [okvqa,ocrvqa,gqa,textvqa]) split by comma')
    parser.add_argument('--each_ds_num', type=str, default=str(EACH_DS_NUM), 
                        help='Number of samples to select from each dataset, split by comma or int')
    parser.add_argument('--output_path', type=str, default="outputs/analyse_ds_debug.jsonl", 
                        help='Output path for the prepared dataset')
    parser.add_argument('--seed', type=int, default=SEED, 
                        help='Random seed for shuffling')
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        print(f"Output path {args.output_path} already exists. Skipping dataset preparation.")
        exit(0)

    datasets = []
    target_dss = args.target_ds.split(',')
    
    each_ds_num_raw = args.each_ds_num.split(',')
    if len(each_ds_num_raw) == 1:
        each_ds_nums = [int(each_ds_num_raw[0])] * len(target_dss)
    else:
        if len(each_ds_num_raw) != len(target_dss):
            raise ValueError(f"Length of each_ds_num ({len(each_ds_num_raw)}) must match length of target_ds ({len(target_dss)})")
        each_ds_nums = [int(num) for num in each_ds_num_raw]

    for ds_name, each_ds_num in zip(target_dss, each_ds_nums):
        if ds_name not in questions_path_map:
            print(f"Warning: Dataset {ds_name} not recognized. Skipping.")
            continue
        ds = prepare_ds(ds_name, each_ds_num, args.seed)
        datasets.extend(ds)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for item in datasets:
            f.write(json.dumps(item) + "\n")
    print(f"Prepared dataset saved to {args.output_path} with total {len(datasets)} samples.")