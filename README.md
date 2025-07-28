# Do Biased Models Have Biased Thoughts?

This paper studies the effect of chain-of-thought prompting, a recent approach that studies the steps followed by the model before it responds, on fairness. To answer our question, we conduct experiments on $5$ popular large language models using fairness metrics to quantify $11$ different biases in the model's thoughts and output. Our results show that the bias in the thinking steps is not highly correlated with the output bias (less than $0.6$ correlation with a $p$-value smaller than $0.001$ in most cases). In other words, unlike human beings, the tested models with biased decisions do not always possess biased thoughts.

## Data
1. BBQ dataset (input) can be found [here](https://github.com/nyu-mll/BBQ/tree/main/data) by Parrish et al., ACL Findings 2022
2. Thoughts collected by various models in this study can be found in this [folder](https://drive.google.com/drive/folders/18OZBv4u3sGquUauykCdytHTTRY1XXs64?usp=sharing)

## Running the code
1. Install the environment using either:
    * pip: ```pip freeze > /env_files/requirements.txt```
    * conda: ```conda env create -f /env_filesenvironment.yml```
2. [Huggingface access token](https://huggingface.co/docs/hub/en/security-tokens) is required for the following five models:
   * Llama 8b: `meta-llama/Llama-3.1-8B-Instruct`
   * Mistral: `mistralai/Mistral-7B-Instruct-v0.3`
   * Phi: `microsoft/Phi-3.5-mini-instruct`
   * Qwen: `Qwen/Qwen2.5-7B-Instruct`
   * Gemma: `google/gemma-2-2B-it`
4. Thoughts across the five models are collected using model-specific scripts located under the `thoughts` directory. Each script is named according to the model it processes (e.g., `gemma.py`, `phi.py`, etc.). Note that the input in these files is the BBQ data.
   * `python {model_name}.py`
6. Run `python biased_labels.py` to collect the bias label of thoughts across all the five models using Llama 70b model (ground truth for this study).
8. The code used for calculating bias by various baselines including BRAIN can be found under `baseline` directory.
  
## 📑 Citation

Please consider citing 📑 our paper if our repository is helpful to your work.
```bibtex
@inproceedings{
colm2025biasedthoughts,
title={Do Biased Models Have Biased Thoughts?},
author={Swati Rajwal, Shivank Garg, Reem Abdel-Salam, Abdelrahman Zayed},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=vDr0RV3590}
}
```
