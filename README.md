# Hierarchical Multi-Label Classification for Software Defects

## Overview

This repository contains the code and resources for the paper, "*Hierarchical Multi-label Classification for Concrete Defects: An Industrial Case Study at Vermeg*." The paper presents a hierarchical multi-label classification model, C-HMCNN, tailored for classifying software defects. It leverages hierarchical taxonomies and includes additional methods (SVM, FFNN, and transformers-based models) for comparative analysis. 

Our approach provides enhanced interpretability and accuracy, specifically in a complex software system with layered defect structures, tested within Vermeg, a software company specializing in financial services solutions.

## Repository Structure

- **C-HMCNN**: Code and model files for the main hierarchical classification model, Coherent Hierarchical Multi-Label Classification Networks (C-HMCNN).
- **Preprocessing**: Scripts for data preprocessing, including feature engineering, label parsing, and data augmentation.
- **Models**: Additional baseline models, including:
  - **SVM**: Support Vector Machine models for binary multi-label classification.
  - **FFNN**: Feedforward Neural Network for flat multi-label classification.
  - **Transformers**: Pre-trained transformer models such as BERT, RoBERTa, and T5 used for comparative analysis.

## Results
C-HMCNN outperformed other models, achieving high F1 scores with both the original and augmented datasets. Hierarchical metrics highlight its robustness in managing multi-level defect categorization, essential for effective defect triaging in complex systems.

## Citation
If you use this work, please cite:
```
@inproceedings{nour2025hierarchical,
  title={Hierarchical Multi-label Classification for Concrete Defects: An Industrial Case Study at Vermeg},
  author={Ahmed Nour, Ilyes Ben Khalifa, Montassar Ben Messaoud, Mohamed Tounsi, Mohamed Wiem Mkaouer},
  booktitle={ICSE SEIP},
  year={2025}
}
```