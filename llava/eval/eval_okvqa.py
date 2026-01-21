import os
import argparse
import json
import re

from llava.eval.vqa import VQA
from llava.eval.vqa_eval import VQAEval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-file", type=str)

    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file,question_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    results = [json.loads(line) for line in open(result_file)]
    pred_list = []
    for result in results: 
        pred_list.append({
            "answer": result['text'],
            "question_id": result['question_id'],
        })

    output_path = os.path.join(os.path.dirname(result_file), 'okvqa_eval.json')
    json.dump(pred_list, open(output_path, 'w'), ensure_ascii=False)
    vqa = VQA(annotation_file,
              question_file)
    results = vqa.loadRes(
        resFile=output_path,
        quesFile=question_file)
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    vqa_acc = vqa_scorer.accuracy
    print(vqa_scorer.accuracy)


    ### 将结果写到文件中 ###
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'result-okvqa.txt')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), vqa_acc['overall']))
    

    ### 将计算结果统一输出到一个汇总的txt文件中 ###
    if args.summary_file is not None: 
        with open(args.summary_file, 'a') as f_sum:
            f_sum.write('\nSamples: {}\nAccuracy on OKVQA: {:.2f}%\n'.format(len(pred_list), vqa_acc['overall']))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.question_file, args.result_file)

