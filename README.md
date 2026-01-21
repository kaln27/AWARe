# AWARe: Activation-Weighted Adaptive REtaining for Mitigating Catastrophic Forgetting in MLLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains the official implementation of the paper **"Activation-Weighted Adaptive REtaining (AWARe)"**.

## 📝 Abstract

> Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, exhibiting strong generalization and reasoning abilities through pre-training on large-scale multimodal corpora. However, when fine-tuning these models on downstream tasks, they suffer from *catastrophic forgetting*—a phenomenon where learning new task-specific knowledge leads to severe degradation of previously acquired capabilities. This forgetting arises because parameter updates optimized for new tasks inadvertently overwrite representations essential for previously learned knowledge, fundamentally limiting the practical deployment of MLLMs.
>
> To address this challenge, we propose **Activation-Weighted Adaptive REtaining (AWARe)**, a method that mitigates catastrophic forgetting in MLLMs by dynamically selecting which parameters to update based on their activation patterns. AWARe uses activation-based importance scores to selectively freeze critical parameters while allowing less important ones to adapt during fine-tuning. Crucially, AWARe operates *without altering the model architecture*, ensuring seamless compatibility with existing inference engines. Experimental results demonstrate that AWARe effectively preserves upstream capabilities while achieving superior performance on downstream tasks compared to existing methods.

## 🚀 Features

- **Dynamic Parameter Selection**: Adaptive freezing of parameters based on activation importance.
- **Architecture-Agnostic**: No structural changes to the model, ensuring compatibility with standard inference pipelines (e.g., vLLM, HuggingFace).
- **Preserves Generalization**: Effectively maintains upstream capabilities (e.g., VQA, Captioning) while fine-tuning on new domains.

## 🛠️ Installation

We use `uv` for dependency management. Please ensure your environment is set up as follows:

1.  **Install dependencies**
    ```bash
    cd AWARe
    # Ensure uv is installed
    pip install uv

    # Sync the environment
    uv sync
    ```

## 📂 Data Preparation

Before running the analysis or training scripts, you need to configure the dataset paths.

1.  **Analysis Data**:
    Modify the `questions_path_map` variable in `AWARe/analyse/prepare_analyse_ds.py` to point to your local dataset files (e.g., OKVQA, OCRVQA, GQA).

    ```python
    # AWARe/analyse/prepare_analyse_ds.py
    questions_path_map = {
        "okvqa": "/path/to/your/instructions/OKVQA/okvqa_val.jsonl",
        # ...
    }
    ```

2.  **Evaluation Data**:
    Update the dataset paths in the evaluation scripts located in `scripts/AWARe/downstream/` and `scripts/AWARe/upstream/`. Ensure variables pointing to image folders and annotation files are correct for your environment.

## 🏃 Usage

The AWARe pipeline consists of three main stages: Analysis, Selection, and Training/Restoration.

### All in one command
```bash
bash run.sh
```

### 1. Analysis
Calculate activation importance scores for the model parameters based on upstream data.

```bash
bash scripts/AWARe/analyse.sh
```

### 2. Selection
Select which nodes/parameters to freeze based on the calculated scores.

```bash
bash scripts/AWARe/select.sh
```

### 3. Training / Restore
Fine-tune the model on downstream tasks while keeping the selected critical parameters frozen.

```bash
bash scripts/AWARe/train.sh
# or
bash scripts/AWARe/restore.sh
```

### 4. Evaluation
Evaluate the model on both upstream (to check forgetting) and downstream tasks.

```bash
# Downstream evaluation (e.g., COCO)
bash scripts/AWARe/downstream/eval_coco.sh

# Upstream evaluation (e.g., GQA)
bash scripts/AWARe/upstream/eval_gqa.sh
```

## 📊 Results

*AWARe significantly outperforms standard fine-tuning and LoRA baselines in preserving upstream performance while maintaining competitive downstream accuracy.*

*(Refer to the paper for detailed result tables and visualization)*

## 🖊️ Citation

If you find this project useful in your research, please cite our paper:

```bibtex
In processing
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
