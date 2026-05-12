# Landmark Classification with Transfer Learning

Achieves ~94% validation accuracy across 51 landmark classes using a fine-tuned ResNet-18 backbone.

---

## Motivation

I'm obsessed with TimeGuessr, a game that gives you photographs and asks you to identify when and where they were taken. This classifier is my first step toward building an AI agent that can actually play it.

---

## Dataset

The model is trained on the Hugging Face dataset:
- [`pemujo/GLDv2_Top_51_Categories`](https://huggingface.co/datasets/pemujo/GLDv2_Top_51_Categories)

This dataset contains ~45k images across 51 landmark classes derived from the Google Landmarks Dataset v2, pre-split 80/20 into train and test sets. Each class corresponds to a unique landmark instance (e.g. Eiffel Tower, Niagara Falls, Edinburgh Castle).

---

## Approach

- Pretrained ResNet-18 backbone (ImageNet weights)
- Final classification layer replaced to output 51 classes
- `layer4` unfrozen for fine-tuning; all other layers frozen
- Image augmentation: random horizontal flip + ImageNet normalization
- LR scheduler: StepLR (step=3, gamma=0.1)
- Uses the dataset's built-in train/test split

---

## Results

- Validation Accuracy: ~94%
- Most classes exceed 90% accuracy by epoch 2
- Weakest class: St. Lawrence Market, Toronto (visually ambiguous indoor market)
- Rapid convergence due to transfer learning — 85% val accuracy after epoch 0

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

```text
src/
  main.py                  # entry point
  data/
    dataset.py             # HuggingFace dataset loading and PyTorch Dataset wrapper
    transforms.py          # image preprocessing and augmentation
  models/
    landmark_model.py      # ResNet-18 model definition
  training/
    train.py               # training loop, evaluation, per-class accuracy
scripts/
  inspect_landmarks.py     # dataset inspection utilities
```

---

## Notes

- Runs on CPU, CUDA, or Apple MPS depending on availability
- Dataset is cached locally (`~/.cache/huggingface/datasets`)
- Trained weights available on Hugging Face Hub: [jblee64/landmark-classification-resnet18](https://huggingface.co/jblee64/landmark-classification-resnet18)
