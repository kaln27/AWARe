import argparse
import json
import os
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="./LLaVA/playground/Instructions_slim/OCRVQA/test_1.json",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
    )
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    output_dir = os.path.dirname(result_file)
    experiment_name = os.path.basename(output_dir)
    annotations = json.load(open(annotation_file))
    annotations = {data["question_id"]: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = []
    for result in results:
        annotation = annotations[result["question_id"]]
        ground_truth = annotation["answer"]
        problem = result["prompt"]
        image = annotation["image"]
        if "Unanswerable" in result["text"]:
            continue

        pred: str = result["text"].lower().replace(" ", "").replace(".", "")
        gt: str = ground_truth.lower()
        if gt == pred:  # all are multiple choice questions or TF questions
            right += 1
        # save the result as jsonl
        pred_list.append(
            dict(
                question=problem,
                pred=result["text"].lower(),
                ground_truth=ground_truth.lower(),
                image=image,
                score=1 if pred == gt else 0,
            )
        )
    accuracy = 100.0 * right / total if total else 0.0
    report = "Samples: {}\nAccuracy: {:.2f}%\n".format(total, accuracy)
    print(report)

    output_file = os.path.join(output_dir, "Result.text")
    with open(output_file, "w") as f:
        f.write(report)
        
    output_file = os.path.join(output_dir, "Result.json")
    with open(output_file, "w") as f:
        for item in pred_list:
            json.dump(item, f)
            f.write("\n")

    if args.summary_file is not None:
        summary_dir = os.path.dirname(args.summary_file)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary_file, "a") as f:
            f.write(f"{experiment_name} -> {report}")


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
