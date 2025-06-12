# 🧪 ToxiMol: A Benchmark for Structure-Level Molecular Detoxification

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2506.12345)

**ToxiMol** is a new large-scale benchmark designed to evaluate structure-level detoxification capabilities of molecular generation models. It provides a rigorous testbed for aligning molecule generation with real-world toxicological requirements, supporting both property control and structural preservation.

---

## 📚 Table of Contents

- [🧬 Overview](#-overview)
- [📂 Dataset Structure](#-dataset-structure)
- [📊 Evaluation](#-evaluation)
- [🧠 Model Evaluation: General-Purpose MLLMs](#-model-evaluation-general-purpose-mllms)
- [🛠 Usage](#-usage)
- [🧑‍🔬 Citation](#-citation)
- [📬 Contact](#-contact)



## 🧬 Overview

**ToxiMol** is the first benchmark specifically designed for evaluating **molecular toxicity repair** — generating structurally valid alternatives to toxic molecules while preserving essential chemical features.

This benchmark provides:
- 🧪 A curated dataset of **560 toxic molecules** across **11 task types**, including functional group preservation, endpoint-specific detoxification, and mechanism-aware edits.
- 🧭 An expert-informed **prompt annotation pipeline**, tailored for general-purpose and chemical-aware models.
- 📊 The **ToxiEval** evaluation framework, offering automated assessment on:
  - Toxicity reduction (Δtox)
  - Structural similarity
  - Chemical validity
  - Drug-likeness

ToxiMol is designed as a benchmark to evaluate the detoxification capabilities of general-purpose Multimodal Large Language Models (MLLMs).  

We systematically test nearly 30 state-of-the-art MLLMs with diverse architectures and input modalities to assess their ability to perform structure-level molecular toxicity repair.



## 📂 Dataset Structure

The **ToxiMol** dataset consists of 560 curated toxic molecules sampled from 12 established toxicity datasets, covering both binary classification and regression tasks across diverse mechanisms:

| Dataset             | Task Type                  | # Molecules | Description                                                                 |
|---------------------|----------------------------|-------------|-----------------------------------------------------------------------------|
| Ames                | Binary Classification      | 50          | Mutagenicity testing                                                        |
| Carcinogens_Lagunin | Binary Classification      | 50          | Carcinogenicity prediction                                                  |
| ClinTox             | Binary Classification      | 50          | Clinical toxicity data                                                      |
| DILI                | Binary Classification      | 50          | Drug-induced liver injury                                                   |
| hERG                | Binary Classification      | 50          | hERG channel inhibition                                                     |
| hERG_Central        | Binary Classification      | 50          | Large-scale hERG database with cardiac safety profiles                      |
| hERG_Karim          | Binary Classification      | 50          | hERG data from Karim et al.                                                 |
| LD50_Zhu            | Regression (log(LD50)<2)   | 50          | Acute toxicity                                                              |
| Skin_Reaction       | Binary Classification      | 50          | Adverse skin reactions                                                      |
| Tox21               | Binary Classification (12 sub-tasks) | 60 | Nuclear receptors & stress response pathways (e.g., ARE, p53, ER, AR)       |
| ToxCast             | Binary Classification (10 sub-tasks) | 50 | Diverse toxicity pathways incl. mitochondrial dysfunction & neurotoxicity  |

Each sample is paired with structural detoxification prompts and evaluation metadata.

You can also access the dataset on Hugging Face:  
👉 [https://huggingface.co/datasets/DeepYoke/ToxiMol-benchmark](https://huggingface.co/datasets/DeepYoke/ToxiMol-benchmark)

## 📊 Evaluation

We propose **ToxiEval**, a multi-dimensional evaluation protocol consisting of the following metrics:

| Metric           | Description                                                                  | Range             | Threshold for Success             |
|------------------|-------------------------------------------------------------------------------|-------------------|-----------------------------------|
| **Safety Score** | Indicates toxicity mitigation, based on TxGemma-Predict classification        | 0–1 or binary     | =1 (binary) or >0.5 (LD50 task)   |
| **QED**          | Drug-likeness score from [0,1]; higher means more drug-like                   | 0–1               | ≥ 0.5                             |
| **SAS**          | Synthetic feasibility; lower scores are better                                | 1–10              | ≤ 6                               |
| **RO5**          | Number of Lipinski rule violations (should be minimal)                        | Integer (≥0)      | ≤ 1                               |

A candidate molecule is considered successfully detoxified **only if it satisfies all four criteria simultaneously**.

---
## 🧠 Model Evaluation: General-Purpose MLLMs

We evaluate **27 general-purpose Multimodal Large Language Models (MLLMs)** on the ToxiMol benchmark to assess their structure-level detoxification capabilities.

These models include:
- **Proprietary MLLMs**: GPT-4o, Gemini-2.5 pro-exp, Claude-3.7 Sonnet Thinking, Grok-2-vision, etc.
- **Open-source MLLMs**: InternVL3.0, Qwen2.5-VL, DeepSeek-VL2, LLaVA-OneVision, etc.


📂 Evaluation scripts and model wrappers are provided in the [`mllm_eval/`](./mllm_eval) directory.---

## 🛠 Usage

```bash
# Clone the repo
git clone https://github.com/your-org/ToxiMol.git
cd ToxiMol

# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation
python eval.py --model MolGPT --dataset ./data/tox_pairs.csv
```

---

## 🧑‍🔬 Citation

If you use this benchmark, please cite:

```bibtex
@article{lin2025breaking,
  title={Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?},
  author={Fei Lin and Ziyang Gong and Cong Wang and Yonglin Tian and Tengchao Zhang and Xue Yang and Gen Luo and Fei-Yue Wang},
  journal={arXiv preprint},
  year={2025},
  note={Paper submitted to arXiv}
}
```

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact [@TengchaoZhang](mailto:zhangtengchao@ieee.org).
