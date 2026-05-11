# Landmark Classification with Transfer Learning

Achieves ~94% validation accuracy across 51 landmark classes using a fine-tuned ResNet-18 backbone.

---

## Overview

This project trains a convolutional neural network to classify images into 51 landmark categories using transfer learning. The project was refactored to simplify the pipeline and make experiments easier to reproduce.

---

## Motivation

I'm obsessed with TimeGuessr, a game that gives you photographs and asks you to identify when and where they were taken. This classifier is my first step toward building an AI agent that can actually play it.

---

## Dataset

The model is trained on the Hugging Face dataset:

- [`pemujo/GLDv2_Top_51_Categories`](https://huggingface.co/datasets/pemujo/GLDv2_Top_51_Categories)

This dataset contains ~36k images across 51 landmark classes derived from the Google Landmarks Dataset v2.

Each class corresponds to a unique landmark instance (identified by `landmark_id`). Since human-readable labels aren’t always available, the task is treated as instance-level classification.

---

## Approach

- Used a pretrained ResNet-18 backbone pretrained on ImageNet
- Replaced the final layer to match 51 landmark classes
- Fine-tuned deeper layers by unfreezing `layer4`
- Applied basic image augmentation (resize + horizontal flip)
- Performed a randomized train/validation split with `train_test_split`

The project loads image data from Hugging Face, applies preprocessing and augmentation, and trains a ResNet-18 model on a train/validation split.

---

## Results

- Validation Accuracy: ~94%
- Train/Validation gap: ~4%
- Consistently strong performance across most classes
- Rapid convergence due to transfer learning

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main
```

---

## Project Structure

```
src/
  main.py
  data/
    dataset.py         # script for downloading and handling the dataset and metadata
    transforms.py      # image-preprocessing script(s)
  models/
    landmark_model.py  # The ResNet-based CNN model
  training/
    train.py           # training + evaluation
scripts/
  inspect_landmarks.py # misc. dataset inspection script
```

---

## Notes

- Runs on CPU, CUDA, or Apple MPS depending on availability  
- Dataset is cached locally (`~/.cache/huggingface/datasets`)
