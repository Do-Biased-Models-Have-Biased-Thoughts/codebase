# 🤔 Do Biased Models Have Biased Thoughts?

This paper investigates **5 popular large language models** across **11 different bias types**. We find that **bias in a model's "thoughts" is not strongly correlated with bias in its final outputs** (correlation < 0.6, *p* < 0.001 in most cases). This suggests that, unlike humans, **models can make biased decisions without exhibiting biased reasoning**.

<figure>
  <img src="2025_COLM.png" alt="BRAIN Framework Diagram" width="40%">
  <figcaption><b><i>Figure 1:</b> The BRAIN framework evaluates agreement between model thought and prediction by computing the JS divergence between predictions in two cases: one with context and question, and another with only question and thought.</i></figcaption>
</figure>

---

## 📂 Dataset

- 🔗 Input [**BBQ dataset**](https://github.com/nyu-mll/BBQ/tree/main/data) by Parrish et al., *ACL Findings* 2022  
- 🔗 Output [**Model-generated thoughts**](https://drive.google.com/drive/folders/18OZBv4u3sGquUauykCdytHTTRY1XXs64?usp=sharing)

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
### 📚 BibTeX
```bibtex
@inproceedings{colm2025biasedthoughts,
  title={Do Biased Models Have Biased Thoughts?},
  author={Swati Rajwal, Shivank Garg, Reem Abdel-Salam, Abdelrahman Zayed},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://openreview.net/forum?id=vDr0RV3590}
}
```
### 🔬 APA Style  
Rajwal, S., Garg, S., Abdel-Salam, R., & Zayed, A. (2025). *Do biased models have biased thoughts?* In *Proceedings of the Second Conference on Language Modeling*. Retrieved from https://openreview.net/forum?id=vDr0RV3590

### 📖 MLA Style  
Rajwal, Swati, Shivank Garg, Reem Abdel-Salam, and Abdelrahman Zayed. "Do Biased Models Have Biased Thoughts?" Proceedings of the Second Conference on Language Modeling, 2025, https://openreview.net/forum?id=vDr0RV3590

---

### ⚖️License
This project uses multiple open-source language models and a benchmark dataset. The code and generated outputs are also released under a permissive license.
<table>
  <thead>
    <tr>
      <th>Component</th>
      <th>Name</th>
      <th>License<br><small><em>(click to read)</em></small></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="vertical-align: middle; text-align: center;">Model</td>
      <td>meta-llama/Llama-3.1-8B-Instruct</td>
      <td><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Community-blue" alt="Community License Agreement"></a></td>
    </tr>
    <tr>
      <td>meta-llama/Meta-Llama-3-70B-Instruct</td>
      <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Community-blue" alt="Community License Agreement"></a></td>
    </tr>
    <tr>
      <td>mistralai/Mistral-7B-Instruct-v0.3</td>
      <td><a href="https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 License"></a></td>
    </tr>
    <tr>
      <td>microsoft/Phi-3.5-mini-instruct</td>
      <td><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License"></a></td>
    </tr>
    <tr>
      <td>Qwen/Qwen2.5-7B-Instruct</td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 License"></a></td>
    </tr>
    <tr>
      <td>google/gemma-2-2B-it</td>
      <td><a href="https://ai.google.dev/gemma/terms"><img src="https://img.shields.io/badge/license-Gemma-orange" alt="Gemma License"></a></td>
    </tr>
    <tr>
      <td>Dataset</td>
      <td>Bias Benchmark for QA (BBQ)</td>
      <td><a href="https://github.com/nyu-mll/BBQ?tab=CC-BY-4.0-1-ov-file"><img src="https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey" alt="CC BY 4.0"></a></td>
    </tr>
    <tr>
      <td>Code</td>
      <td>This Repository</td>
      <td><a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a></td>
    </tr>
  </tbody>
</table>
