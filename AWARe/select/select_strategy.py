from typing import Dict
from tqdm import tqdm
import numpy as np
from typing import DefaultDict

import torch


def select_freezable_nodes_balanced(
    analysis: Dict[str, torch.Tensor], quota: float
) -> Dict[str, list[int]]:
    results = {}
    print("Selecting layer-balanced nodes ...")
    for module in analysis:
        _analysis = analysis[module].numpy()
        num_nodes = _analysis.shape[0]
        num_freezables = int(num_nodes * quota / 100)

        # Select nodes with highest analysis score
        select_indices = np.argsort(_analysis)[::-1][:num_freezables]
        results[module] = sorted(select_indices.tolist())

    return results


def select_freezable_nodes_global_highest(
    analysis: Dict[str, torch.Tensor], quota: float
) -> Dict[str, list[int]]:
    results = {}
    print("Selecting global-highest nodes ...")
    all_scores = []

    for module_name, scores in analysis.items():
        scores_np = scores.numpy()
        for idx, val in enumerate(scores_np):
            all_scores.append({
                'module': module_name,
                'index': idx,
                'score': val
            })

    total_nodes = len(all_scores)
    num_to_select = int(total_nodes * quota / 100)

    all_scores.sort(key=lambda x: x['score'], reverse=True)

    selected_nodes = all_scores[:num_to_select]

    results = {module: [] for module in analysis.keys()}
    for node in selected_nodes:
        results[node['module']].append(node['index'])

    for module in results:
        results[module].sort()

    return results