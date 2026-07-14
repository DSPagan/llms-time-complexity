# Estimating Time Complexity with Large Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSPagan/llms-time-complexity/blob/main/notebooks/fine_tuning.ipynb)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)

Undergraduate thesis (TFG) — **Bachelor's Degree in Mathematics, University of Alicante (2024–2025)**.

This repository contains the code, dataset, experiments, and thesis for a study on whether large language models (LLMs) can estimate the **time complexity of algorithms without executing them**.

> 🌐 **Language note:** the code and this README are written in English for wider reach. The full thesis — [`thesis/tfg-complexity.pdf`](thesis/tfg-complexity.pdf) — is written in **Spanish**, as it was the language of the degree. An English abstract is included below so the work is fully understandable without reading the Spanish document.

## 🧠 Overview

Estimating the time complexity of a program traditionally requires either empirical testing with inputs of varying size or manual step-counting. This project explores a different route: using an LLM to infer the worst-case time complexity directly from source code, framed as a 7-class classification problem over `O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`, `O(n³)`, and exponential.

Three approaches are compared on the same model — **Llama 3.1 8B Instruct**, quantized to 4 bits — to measure how much task-specific adaptation helps:

- **Zero-shot** — the model is asked directly, with prompt engineering.
- **Few-shot** — worked examples are added to the prompt (in-context learning).
- **Fine-tuning** — the model is fine-tuned on labelled examples using **QLoRA**.
- **Chain-of-thought** *(beyond the thesis)* — the model reasons step by step before answering.

## 📄 Abstract

> This work explores the use of large language models (LLMs) to automatically estimate the computational complexity of algorithms without executing them or performing manual analysis. It includes a theoretical review of the Transformer architecture and key concepts such as fine-tuning, quantization, and in-context learning. The performance of the `Llama 3.1 8B Instruct` model, quantized to 4 bits, is evaluated using three approaches — *zero-shot*, *few-shot*, and fine-tuning with QLoRA — based on the `CodeComplex` dataset. Results show that LLMs can provide reasonable complexity estimates even without specific training, but reach optimal performance after fine-tuning (up to **91.2% accuracy**). This study highlights the potential of LLMs as support tools for algorithmic analysis, while also acknowledging limitations related to code ambiguity, generalization, and computational resource constraints.

## 📊 Results

> ⏳ **Being refreshed.** The numbers and figures below are the results reported in the thesis. The data pipeline has since been made fully reproducible (regenerated from the original CodeComplex snapshot), a chain-of-thought prompt was added, and every experiment now runs under **5-fold cross-validation**, so these will be re-measured (as mean ± std) on the current setup.

The three approaches were evaluated on a held-out test set with **accuracy** and **macro F1-score**. Fine-tuning is by far the largest driver of performance:

| Approach                | Best configuration    | Accuracy   | Macro F1   |
| ----------------------- | --------------------- | ---------- | ---------- |
| Zero-shot               | prompt 2              | 45.0%      | 46.8       |
| Few-shot                | in-context examples   | 49.7%      | 51.8       |
| **Fine-tuned (QLoRA)**  | **2 epochs**          | **91.2%**  | **91.8**   |

In-context examples give a modest lift over zero-shot, but fine-tuning roughly **doubles accuracy**. The fine-tuning gain is progressive with training: 66.2% (60 steps) → 90.1% (1 epoch) → 91.2% (2 epochs).

The confusion matrices below show the effect clearly — predictions move from scattered (zero-shot) to a strong diagonal (fine-tuned):

| Zero-shot (45.0%) | Fine-tuned with QLoRA, 2 epochs (91.2%) |
| :---: | :---: |
| ![Zero-shot confusion matrix](figures/CM_zeroshot_v2.png) | ![Fine-tuned confusion matrix](figures/CM_QLoRA_v3.png) |

After fine-tuning, the most frequent remaining errors are confusions between adjacent complexity classes (e.g. `O(1)` vs `O(log n)`, `O(n log n)` vs `O(n)`).

## 🗂️ Dataset

Experiments use the **[CodeComplex](https://doi.org/10.48550/arXiv.2401.08719)** dataset (Baik et al., 2024): Python snippets labelled with their worst-case time complexity across 7 classes (`O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n²)`, `O(n³)`, exponential).

The pipeline starts from the original CodeComplex snapshot ([`data/data.jsonl`](data/data.jsonl)) and is fully reproducible: [`src/prepare_data.py`](src/prepare_data.py) de-duplicates by source code and splits the data into stratified 5-fold cross-validation partitions in memory (`load_clean` + `stratified_folds`). The dataset is redistributed for reproducibility and remains subject to its original license — see [License](#-license).

## 📂 Repository structure

```text
.
├── data/          # the original CodeComplex snapshot (data.jsonl)
├── figures/       # Confusion matrices from the experiments
├── notebooks/     # fine_tuning.ipynb and in_context_learning.ipynb (Colab)
├── src/           # prepare_data, load_model, prompts, train_model, evaluate
├── thesis/        # LaTeX source and compiled PDF of the thesis (Spanish)
├── outputs/       # Fine-tuned adapters and inference results (generated at runtime, not tracked)
├── requirements.txt       # How to install the stack
├── requirements-lock.txt  # Exact versions verified to work
├── LICENSE
└── README.md
```

## 🛠️ Installation

Requires **Python 3.10–3.12** and a **CUDA-enabled GPU** (on Google Colab a CUDA-enabled PyTorch is already present). Install the pinned stack:

```bash
pip install -r requirements-lock.txt
```

The easiest path is to run the notebooks in Colab (badges above) — they clone the repo and install from this lock automatically.

> **Why pinned, not latest:** Unsloth moves fast enough to break between releases (a newer version stopped resolving the 4-bit model repo mid-project). [`requirements-lock.txt`](requirements-lock.txt) pins the exact versions verified to work end to end, so **every experiment runs on the same stack** — which is what keeps the results comparable across notebooks.

## 🚀 Usage

Two Colab notebooks reproduce the experiments end to end under **5-fold cross-validation** (each clones the repo, builds the folds, and runs on a GPU):

- [`notebooks/in_context_learning.ipynb`](notebooks/in_context_learning.ipynb) — zero-shot (2 prompts), few-shot, and chain-of-thought on the base model.
- [`notebooks/fine_tuning.ipynb`](notebooks/fine_tuning.ipynb) — QLoRA fine-tuning, one fold per session.

To use the pieces programmatically, the modules in [`src/`](src/) compose:

```python
from src.load_model import load_model
from src.prepare_data import load_clean, stratified_folds

model, tokenizer = load_model()          # Llama 3.1 8B Instruct, 4-bit
folds = stratified_folds(load_clean())   # 5 stratified CV folds

# The notebooks compose these with src.train_model and src.evaluate
# into the full zero-/few-shot/CoT and QLoRA cross-validation loops.
```

## 🔬 Methodology

1. **Dataset** — CodeComplex snippets annotated with time complexity.
2. **Prompt engineering** — designing prompts that elicit accurate zero-shot predictions.
3. **In-context learning** — adding worked examples (few-shot).
4. **Fine-tuning** — training with QLoRA on algorithm–complexity pairs.
5. **Evaluation** — accuracy and macro F1 under **5-fold cross-validation** (reported as mean ± std), plus confusion-matrix analysis.

## 📚 Thesis

The full memoria (in Spanish) is available at [`thesis/tfg-complexity.pdf`](thesis/tfg-complexity.pdf), with LaTeX source in the same folder. It covers the theoretical background (Transformers, fine-tuning, quantization, in-context learning) and a detailed analysis of every experiment summarized above.

<details>
<summary>Key references</summary>

- Baik, S.-Y., Hahn, J., Kim, J., Jeon, M., Han, Y.-S., & Ko, S.-K. (2024). [CodeComplex: Dataset for worst-case time complexity prediction](https://doi.org/10.48550/arXiv.2401.08719). *arXiv:2401.08719*.
- Touvron, H., et al. (2024). [The Llama 3 herd of models](https://arxiv.org/abs/2407.21783). *arXiv:2407.21783*.
- Vaswani, A., et al. (2017). [Attention is all you need](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). *NeurIPS 30*.
- Brown, T. B., et al. (2020). [Language models are few-shot learners](https://arxiv.org/abs/2005.14165). *NeurIPS 33*, 1877–1901.
- Hu, E. J., et al. (2021). [LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685). *arXiv:2106.09685*.

The complete bibliography is in the thesis.

</details>

## 🧑‍💻 Author

**Daniel Sánchez Pagán**
Bachelor's Degree in Mathematics — University of Alicante
Academic Year 2024–2025

## 📄 License

- **Code** (`src/`, `notebooks/`) — [MIT](LICENSE).
- **Thesis** (`thesis/`) — © 2025 Daniel Sánchez Pagán, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Dataset** (`data/`) — derived from [CodeComplex](https://doi.org/10.48550/arXiv.2401.08719); subject to the terms of the original dataset.
