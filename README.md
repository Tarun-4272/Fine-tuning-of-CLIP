# CLIP Model Fine-Tuning on Food Image Caption Dataset

This project demonstrates the fine-tuning of the CLIP model on a custom dataset consisting of food images paired with textual captions. The goal is to train the CLIP model to learn a shared representation between images and their corresponding text descriptions.

## Table of Contents
- [Introduction](#introduction)
- [Project Setup](#project-setup)
- [Dataset](#dataset)
- [Training Process](#training-process)
  - [Data Augmentation](#data-augmentation)
  - [Model Setup](#model-setup)
  - [Custom Loss Function](#custom-loss-function)
  - [Training and Validation Loop](#training-and-validation-loop)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Files in the Repository](#files-in-the-repository)
- [Conclusion](#conclusion)

## Introduction
This project involves fine-tuning the CLIP model (clip-vit-base-patch32) using a dataset of food images and their corresponding captions. The CLIP model is a powerful model that learns to understand images and text in a shared latent space. Fine-tuning is performed to adapt the model's pre-trained features to this specific task.

## Project setup
To set up the project, ensure that you have the following installed:
- Python 3.6+
- PyTorch
- Hugging Face transformers and datasets libraries
- torchvision for data augmentation
- tqdm for progress tracking

## Dataset

The dataset used in this project is `zmao/food_img_caption_small`, an open-source dataset containing pairs of food images and textual captions. The dataset is loaded using the `datasets` library and is split into training and validation sets for model evaluation.

## Training Process

### Data Augmentation

To improve model generalization, several data augmentation techniques are applied to the images:

- Randomly resizing and cropping images to 224x224 pixels.
- Randomly flipping images horizontally.
- Adjusting brightness, contrast, saturation, and hue to introduce variability.

### Model Setup

The CLIP model (`clip-vit-base-patch32`) is loaded with pre-trained weights. The model is trained using the AdamW optimizer, which includes weight decay to prevent overfitting. A learning rate scheduler (ReduceLROnPlateau) is used to reduce the learning rate if the validation loss stops improving.

### Custom Loss Function

A custom contrastive loss function is used, which computes cross-entropy loss between image logits and text logits, encouraging the model to learn a shared representation between images and text.

### Training and Validation Loop

The training process runs for a maximum of 20 epochs. The model's performance is monitored on the validation set, and early stopping is implemented to prevent overfitting. The best-performing model is saved for later use.

## Model Evaluation

The model is evaluated using the validation loss, which measures how well the model is performing on unseen data. If the validation loss improves, the model is saved as the best model. The training process is stopped early if the model's performance does not improve over several epochs.

## Results

After training, the model's performance is assessed based on the average training and validation losses across epochs. The best model is saved and can be used for further testing or deployment.

## Files in the Repository

- `train.py`: The main script that includes all the steps required to fine-tune the CLIP model.
- `best_model.pt`: The saved state of the best-performing model during training.
- `README.md`: This documentation file.

## Conclusion

This project demonstrates how to fine-tune the CLIP model on a custom dataset of food images and captions. By using data augmentation, a custom loss function, and careful training, the model is adapted to the specific task, improving its performance on image-to-text and text-to-image tasks within the food domain.



