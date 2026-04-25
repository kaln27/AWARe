import os
import json
import random
import ast
from datasets import load_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Prepare MMMU dataset for analysis.")
parser.add_argument("--output_dir", "-o",  type=str, default="outputs/mmmu_sample", help="Directory to save the prepared dataset.")
parser.add_argument("--num_samples_per_subtask", "-n", type=int, default=20, help="Number of samples to extract from each sub-task.")
args = parser.parse_args()

# Configuration
seed = 42
random.seed(seed)
ds_path = '/mnt/datasets/MMMU/MMMU'
output_dir = args.output_dir
output_jsonl = os.path.join(output_dir, 'mmmu_sample.jsonl')
output_image_folder = os.path.join(output_dir, 'images')

if os.path.exists(output_image_folder):
    print(f"MMMU dataset already prepared. Skipping sampling.")
    exit(0)  # Skip if already exists
os.makedirs(output_image_folder, exist_ok=True)

sub_task = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 
    'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 
    'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

results = []

print(f"Sampling 20 examples from each of {len(sub_task)} sub-tasks...")

TEMPLAT = """\
{question}
Here are some options:
{options}
Please provide your answer based on the options above.
""".strip()

for name in tqdm(sub_task):
    # Determine split (usually validation for dev/debugging)
    try:
        # Attempt to load validation set first
        ds = load_dataset(ds_path, name=name, split='test')
    except Exception as e:
        print(f"Could not load validation for {name}, trying test. Error: {e}")
        try:
            ds = load_dataset(ds_path, name=name, split='test')
        except Exception as e2:
            print(f"Failed to load {name}. Error: {e2}")
            continue

    ds.shuffle(seed=seed)

    count = 0
    for item in ds:
        # Make sure only single-image items are processed
        if item['image_2'] is not None:
            continue

        image = item['image_1']
        if image.mode != 'RGB':
            try:
                image = image.convert('RGB')
            except Exception as e:
                print(f"Failed to convert image for {name} {item['id']}: {e}")
                continue
            
        file_name = f"{name}_{item['id']}.jpg"
        image_path = os.path.join(output_image_folder, file_name)
        image.save(image_path)
        
        # Format Text
        question = item['question'].replace('<image_1>', '').strip()
        options = item['options'] # usually a string representation of list like "['(A) ..', '(B) ..']"
        
        # Parse options if it's a string
        if isinstance(options, str):
            try:
                options_list = ast.literal_eval(options)
                # Re-format nicely usually
                options_str = " ".join(options_list) 
            except:
                options_str = options
        else:
             # If it's already a list (depending on dataset version)
             options_str = " ".join(options)

        # Construct the prompt text
        # Using a standard format: Question + Options + Instruction
        text = TEMPLAT.format(question=question, options=options_str)

        entry = {
            "question_id": item['id'],
            "image": image_path,
            "text": text,
            "category": name
        }
        
        results.append(entry)
        count += 1
        if count >= args.num_samples_per_subtask:
            break

# Write outputs
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for entry in results:
        f.write(json.dumps(entry) + '\n')

print(f"Finished. Saved {len(results)} items to {output_jsonl}")
print(f"Images saved to {output_image_folder}")