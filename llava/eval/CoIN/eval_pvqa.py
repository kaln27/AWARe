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
    for result in results:
        annotation = annotations[result["question_id"]]
        ground_truth = annotation["answer"]
        if "Unanswerable" in result["text"]:
            continue
        # if result['text'].lower() == ground_truth.lower(): # TODO: need to check which rules to use
        #     right += 1
        if ground_truth.lower() in result["text"].lower():
            right += 1

    accuracy = 100.0 * right / total if total else 0.0
    report = "Samples: {}\nAccuracy: {:.2f}%\n".format(total, accuracy)
    print(report)

    if output_dir is not None:
        output_file = os.path.join(output_dir, "Result.text")
        with open(output_file, "w") as f:
            f.write(report)

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
