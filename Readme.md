# Generative Pretrained Embedding and Hierarchical Representation to Unlock Human Rhythm in Activities of Daily Living

Welcome to the official GitHub repository for our article titled "Generative Pretrained Embedding and Hierarchical Representation to Unlock Human Rhythm in Activities of Daily Living". This repository contains the code, figures, and additional resources related to our research.

![GPTHAR Neural Network Architecture](Figures/gpthar_architecture.jpg)

## Abstract
Our research presents a novel approach to understanding human rhythm in daily activities using the Generative Pretrained Embedding and Hierarchical Representation (GPTHAR) method. Through this, we seek to unlock new dimensions in activity recognition and bring forward a nuanced model that accommodates various real-world scenarios.

## Datasets
To assess the robustness and versatility of our approach, we utilized the following datasets from the [CASAS collection](https://casas.wsu.edu/datasets/)

- **Aruba Dataset**: A dataset used to validate our model's proficiency in scenarios with a single resident.
- **Milan Dataset**: Used to probe the robustness of our model in the face of potential disturbances caused by pets or sensor issues.
- **Cairo Dataset**: Assesses the adaptability of our model in environments with multiple residents and potential overlapping activities.

All these datasets are part of the CASAS collection, and they originate from the homes of volunteers, collected over several months, presenting unbalanced classes.

### Details of the Selected Datasets

| **Dataset**           | **Aruba** | **Milan** | **Cairo** |
|-----------------------|-----------|-----------|-----------|
| Residents             | 1         | 1 + pet   | 2 + pet   |
| Number of Sensors     | 39        | 33        | 27        |
| Number of Activities  | 12        | 16        | 13        |
| Number of Days        | 219       | 82        | 56        |


## Results
Our proposed GPTHAR algorithm showed significant improvements over existing methods. The results will be updated soon.

| Method          | Metric 1 (%) | Metric 2 (%) | ... |
|-----------------|--------------|--------------|-----|
| Existing Method | xx.x         | xx.x         | ... |
| GPTHAR          | xx.x         | xx.x         | ... |

## Installation
