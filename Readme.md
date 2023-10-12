# Generative Pretrained Embedding and Hierarchical Representation to Unlock Human Rhythm in Activities of Daily Living

Welcome to the official GitHub repository for our article titled "Generative Pretrained Embedding and Hierarchical Representation to Unlock Human Rhythm in Activities of Daily Living". This repository contains the code, figures, and additional resources related to our research.

![GPTHAR Neural Network Architecture](Figures/gpthar_architecture.jpg)

## Abstract
Our research presents a novel approach to understanding human rhythm in daily activities using the Generative Pretrained Embedding and Hierarchical Representation (GPTHAR) method. Through this, we seek to unlock new dimensions in activity recognition and bring forward a nuanced model that accommodates various real-world scenarios.

## Datasets
To assess the robustness and versatility of our approach, we utilized the following datasets from the [CASAS collection](https://casas.wsu.edu/datasets/):

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


### Results from Final Test Set

|                               | **Aruba**        | **Aruba (std)** | **Milan**        | **Milan (std)** | **Cairo**        | **Cairo (std)** |
|-------------------------------|------------------|-----------------|------------------|-----------------|------------------|-----------------|
| **ELMoAR (Window 60)**        | 84.76%           | 0.32            | 68.51%           | 1.02            | 69.12%           | 1.76            |
| **GPTAR (8 Heads 3 Layers)**  | 85.18%           | **0.22**        | 68.55%           | **1.00**        | 73.33%           | 2.08            |
| **ELMoHAR**                   | 88.22%           | 3.79            | 73.91%           | 1.31            | 74.75%           | 1.98            |
| **GPTHAR**                    | 88.55%           | 3.83            | 76.87%           | 1.61            | 83.60%           | 1.90            |
| **ELMoHAR + Time encoding**   | 89.71%           | 0.69            | 77.84%           | 3.24            | 81.87%           | 2.26            |
| **GPTHAR + Time encoding**    | **90.10%**       | 1.05            | **79.22%**       | 1.27            | **86.74%**       | **1.16**        |


## Installation
