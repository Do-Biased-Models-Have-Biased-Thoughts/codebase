# Do Biased Models Have Biased Thoughts?

This paper studies the effect of chain-of-thought prompting, a recent approach that studies the steps followed by the model before it responds, on fairness. To answer our question, we conduct experiments on $5$ popular large language models using fairness metrics to quantify $11$ different biases in the model's thoughts and output. Our results show that the bias in the thinking steps is not highly correlated with the output bias (less than $0.6$ correlation with a $p$-value smaller than $0.001$ in most cases). In other words, unlike human beings, the tested models with biased decisions do not always possess biased thoughts.

## Data
1. BBQ dataset (input) can be found [here](https://github.com/nyu-mll/BBQ/tree/main/data) by Parrish et al., ACL Findings 2022
2. Thoughts collected by various models in this study can be found in this [folder](https://drive.google.com/drive/folders/18OZBv4u3sGquUauykCdytHTTRY1XXs64?usp=sharing)

## Running the code
1. Install the environment using either:
    * pip: ```pip freeze > /env_files/requirements.txt```
    * conda: ```conda env create -f /env_filesenvironment.yml```
2. Thoughts collection:
3. Biased label collection:
4. Baselines:   
  
## 📑 Citation

Please consider citing 📑 our paper if our repository is helpful to your work.
```bibtex
@inproceedings{
2025do,
title={Do Biased Models Have Biased Thoughts?},
author={Swati Rajwal, Shivank Garg, Reem Abdel-Salam, Abdelrahman Zayed},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=vDr0RV3590}
}
```
