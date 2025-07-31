# 🤔 Do Biased Models Have Biased Thoughts?

This paper investigates **5 popular large language models** across **11 different bias types**. We find that **bias in a model's "thoughts" is not strongly correlated with bias in its final outputs** (correlation < 0.6, *p* < 0.001 in most cases). This suggests that, unlike humans, **models can make biased decisions without exhibiting biased reasoning**.

<figure>
  <img src="2025_COLM.png" alt="BRAIN Framework Diagram" width="40%">
  <figcaption><b><i>Figure 1:</b> BRAIN framework for evaluating agreement between model thought and prediction. BRAIN computes the JS divergence between the predictions for two cases: once when feeding the context and question, and another when feeding the question and thought.</i></figcaption>
</figure>

---

## 📂 Dataset

- 🔗 Input [**BBQ dataset**](https://github.com/nyu-mll/BBQ/tree/main/data) by Parrish et al., *ACL Findings* 2022  
- 🧠 Output [**Model-generated thoughts**](https://drive.google.com/drive/folders/18OZBv4u3sGquUauykCdytHTTRY1XXs64?usp=sharing)

---

## ⚙️ How to Run

### 1⃣ Set Up Environment

- **Using pip:**
  ```bash
  pip install -r env_files/requirements.txt
  ```

- **Using conda:**
  ```bash
  conda env create -f env_files/environment.yml
  ```

### 2⃣ Required Access

- 🔐 You will need a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) for the following models:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `meta-llama/Meta-Llama-3-70B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `microsoft/Phi-3.5-mini-instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
  - `google/gemma-2-2B-it`

### 3⃣ Collect Model Thoughts

Each model has a corresponding script under the `thoughts/` directory:

```bash
# Example for Gemma
python thoughts/gemma.py
```

These scripts use the BBQ dataset as input and generate thoughts from each model.

### 4⃣ Annotate Bias in Thoughts

Use the Llama 70b model as ground truth model:

```bash
python biased_labels.py
```

This assigns bias labels to thoughts from all models.

### 5⃣ Run Baselines

Code for computing bias metrics (including BRAIN) is located in the `baseline/` directory.

---

## 📁 Citation

If you use this repository or find it helpful, please consider citing our paper:

```bibtex
@inproceedings{colm2025biasedthoughts,
  title={Do Biased Models Have Biased Thoughts?},
  author={Swati Rajwal, Shivank Garg, Reem Abdel-Salam, Abdelrahman Zayed},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://openreview.net/forum?id=vDr0RV3590}
}
```

---
