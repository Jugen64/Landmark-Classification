# Landmark Classification with Transfer Learning

Classifies photos into 51 famous landmark categories using a fine-tuned ResNet-18. Achieves **95.2% validation accuracy** on the [GLDv2 Top 51 dataset](https://huggingface.co/datasets/pemujo/GLDv2_Top_51_Categories).

Built as a first step toward an AI agent that can play [TimeGuessr](https://timeguessr.com).

---

## Quickstart

```bash
pip install -r requirements.txt
python -m src.main
```

## Inference

Download weights and run predictions on a single image:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jblee64/landmark-classification-resnet18", filename="best_model.pth", local_dir=".")
```

```bash
python inference.py path/to/image.jpg
```

Prints top 5 predicted landmarks with confidence scores.

---

## Model

- ResNet-18 pretrained on ImageNet, `layer4` fine-tuned
- StepLR scheduler (step=3, gamma=0.1)
- Trained weights: [jblee64/landmark-classification-resnet18](https://huggingface.co/jblee64/landmark-classification-resnet18)

## Notes

- Runs on CPU, CUDA, or Apple MPS
- Dataset cached locally via HuggingFace
