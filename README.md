# Cascaded Selective Evaluation — Random Split Automation

## Background

This project is part of an ongoing research effort to improve the reliability and coverage of automated LLM-based evaluation systems.

Modern NLP pipelines often use large language models as evaluators. However, a single model may fail to confidently evaluate all samples — some inputs are too ambiguous or difficult. The **cascaded selective evaluation** framework addresses this by chaining multiple models in sequence: if the first model is not confident enough on a sample, it passes the sample to the next model in the cascade, and so on. This allows the system to maintain high accuracy while maximizing the proportion of samples that are evaluated (coverage).

The key challenge is calibration. Each model in the cascade needs a calibration dataset to set its confidence threshold (λ). If the calibration set is not representative of the test set, the resulting accuracy and coverage can be unreliable. To address this, we use a **random split strategy**: instead of using a fixed calibration/test split, we repeatedly and randomly re-split the full dataset and run the evaluation each time, collecting results across many trials.

## This Repository

This repo contains two scripts developed during **Phase 2** of the research project, focused on improving coverage using the cascaded method:

- `example_apply_decision_rule.py` — The original example script demonstrating how the cascaded classifier is applied across all model permutations with a fixed calibration/test split.
- `run_random_splits.py` — An automation script that randomly re-splits the full dataset (580 samples) into calibration (400) and test (180) sets across 100 trials, runs the cascaded evaluation for all 24 model permutations per trial, and records results sorted by coverage.

## How to Run

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the automation script
python run_random_splits.py
```

Results are saved as a `.csv` file with accuracy, coverage, evaluator composition, and the sample ID order for each split.

---

## 背景

本项目是一个研究项目，目标是提升基于大语言模型（LLM）的自动化评估系统的可靠性与覆盖率。

在自动化评估流程中，单一模型往往无法对所有样本给出高置信度的判断。**级联选择性评估（Cascaded Selective Evaluation）** 方法通过将多个模型串联起来解决这一问题：当第一个模型对某个样本的置信度不足时，该样本会被传递给下一个模型，依此类推。这种方式可以在保证高准确率的同时，尽可能提高被评估样本的比例（即覆盖率）。

核心挑战在于校准（calibration）。每个模型需要一个校准数据集来确定其置信度阈值（λ）。如果校准集与测试集的分布不一致，得出的准确率和覆盖率结果会产生较大偏差。为解决这一问题，我们采用**随机拆分策略**：不使用固定的校准/测试集划分，而是对完整数据集进行多次随机重新划分，分别运行评估并收集结果。

## 本仓库内容

本仓库包含研究项目**第二阶段**开发的两个脚本，专注于通过级联方法提升覆盖率：

- `example_apply_decision_rule.py` — 原始示例脚本，展示如何在固定校准/测试集划分下，对所有模型排列组合运行级联分类器。
- `run_random_splits.py` — 自动化脚本。将完整数据集（580条样本）随机拆分为校准集（400条）和测试集（180条），共运行100次，每次对24种模型排列组合分别评估，结果按覆盖率从高到低排序保存为CSV文件。

## 运行方式

```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行自动化脚本
python run_random_splits.py
```

结果将保存为 `.csv` 文件，包含每次拆分的准确率、覆盖率、评估器组成及样本ID顺序。
