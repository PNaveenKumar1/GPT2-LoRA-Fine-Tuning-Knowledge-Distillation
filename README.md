# LoRA Fine-Tuning & Knowledge Distillation

## Overview

This project involves fine-tuning GPT2 models using the LoRA (Low Rank Adaptation) method and applying Knowledge Distillation to transfer knowledge from a fine-tuned GPT2 model to a smaller RNN-based model. The goal is to achieve efficient classification on the CoLA dataset using parameter-efficient fine-tuning and distillation techniques.

## Setup

1. Ensure you have the latest stable version of PyTorch installed along with the `transformers` library from HuggingFace:
    ```bash
    pip install torch transformers
    ```

## Data

The dataset used in this project is the CoLA (Corpus of Linguistic Acceptability) dataset. It consists of sentences labeled as acceptable or unacceptable, and we aim to classify these sentences using fine-tuned models.

- **Dataset URL**: [CoLA Dataset](https://nyu-mll.github.io/CoLA)
- **Format**: 
    - Column 1: Sentence source
    - Column 2: Acceptability judgment (0=unacceptable, 1=acceptable)
    - Column 3: Acceptability judgment from the author

## Problem 0: Generation

- **Task**: Ensure that all libraries are set up correctly and GPT2 can generate text sequences.
- **Implementation**:  `run.py` to load the GPT2 model and generate text. execute the code and generate the sequences using the GPT2 model.

## Problem 1: Low Rank Adaptation (LoRA)

### LoRA Fine-Tuning

- **LoRA Overview**: LoRA is a parameter-efficient fine-tuning method where instead of updating the entire weight matrix of a pre-trained model, two smaller matrices (L and R) are used to approximate the weight updates.
    - Formula: \( W_0 + \Delta W = W_0 + LR \)
    - \( W_0 \) is the original weight matrix, and \( \Delta W = LR \) represents the low-rank update.
    - The rank \( r \) is a hyperparameter, which was set to 4 based on experiments.

### Training Strategy

- **GPT2 Model Variants**: Both GPT2-base and GPT2-medium models were fine-tuned on the CoLA dataset.
- **Hyperparameters**:
    - Learning rate: 1e-3
    - Batch size: 128
    - Epochs: 10 (with early stopping based on validation accuracy)
    - LoRA rank: 4
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy loss

### Results

- **GPT2-Base**:
    - Number of Parameters: 125.03M
    - Number of Trainable Parameters: 0.68M
    - Reduction: 99.46%
    - Maximum Accuracy: 79.70% on the CoLA validation set.
    - **Plots**: Losses and accuracies are plotted for epochs and can be found in the `plots/` folder.

- **GPT2-Medium**:
    - Number of Parameters: 356.40M
    - Number of Trainable Parameters: 1.80M
    - Reduction: 99.50%
    - Maximum Accuracy: 82.92% on the CoLA validation set.
    - **Plots**: Losses and accuracies are plotted for epochs and can be found in the `plots/` folder.

## Problem 2: Knowledge Distillation (KD)

### Knowledge Distillation Overview

- **Objective**: Transfer knowledge from a large teacher model (GPT2) to a smaller student model (DistilRNN) to reduce model size while retaining performance.
- **Temperature**: A hyperparameter that controls the smoothness of the output distribution from the teacher model during distillation. Set to 2.

### DistilRNN Architecture

- **Layers**:
    - Embedding layer: Maps input tokens to a 768-dimensional embedding space.
    - RNN layer: Processes the sequence of embeddings with 768 hidden units.
    - Linear layer: Performs final classification into binary classes.
- **Training Hyperparameters**:
    - Learning rate: 1e-3
    - Batch size: 128
    - Epochs: 5
- **Loss Function**: Knowledge distillation loss, combined with cross-entropy loss.

### Results

- **DistilRNN (with KD)**:
    - Maximum Accuracy: 72.45% on the CoLA validation set.
    - **Plots**: Losses and accuracies are plotted for epochs and can be found in the `plots/` folder.

- **DistilRNN (without KD)**:
    - Maximum Accuracy: 69.45% on the CoLA validation set.
    - **Plots**: Losses and accuracies are plotted for epochs and can be found in the `plots/` folder.

### Comparison

- The student model with Knowledge Distillation showed a 3% improvement in accuracy compared to the model trained without distillation, demonstrating the effectiveness of knowledge transfer.

1. **Code**: The repository contains all code for LoRA fine-tuning and Knowledge Distillation.
2. **Report**: Detailed explanations and results are available in the report `Report_<22915>.pdf`.
3. **Plots**: Training plots for accuracy and loss can be found in the `plots/` folder.

---

This README provides an overview of the approaches and results for this project. You can find the detailed report and code implementation in this repository.
