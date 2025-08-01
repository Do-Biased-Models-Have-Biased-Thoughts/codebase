#  Confidence Score as a Proxy for Bias

This method quantifies **bias** in a model's thoughts by measuring how confidently an external classifier predicts a **biased answer** using those thoughts as input.

---

##  Method Overview

We define the **confidence score** as the classifier’s probability output:

> **p(y|x)**

Where:
- **x** = the model's thought process or explanation
- **y** = one of the multiple-choice answer options (specifically, the biased one)

A **high p(y|x)** indicates that the model’s thoughts contain implicit cues or assumptions that lead strongly to a specific — and potentially biased — answer.


##  Implementation Steps

### 1. Train a Confidence Classifier usinfg `confidence_score.py`

Fine-tune a powerful model (e.g., `microsoft/deberta-v3-large`) using:

- **Input**: Model's thought/explanation
- **Target**: The original model's selected answer (as a class label)

This trains the classifier to learn how thoughts map to answer choices.

---

### 2. Inference for Bias Scoring

At test time:

- Feed a new **thought explanation** into the classifier
- Measure the **softmax probability** for the **biased answer option**

This probability becomes the **confidence score**.

---

##  Bias Quantification

Use the confidence score **p(biased | x)** as a proxy for bias.

Examples:
- `p(correct class | x) ≥ 0.7` → Strong evidence of unbias
- `p(biased | x) ≈ 0.5` → Unclear or neutral or bias
- `p(biased | x) ≤ 0.5` → Strong bias signal

---
