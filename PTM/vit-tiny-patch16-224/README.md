---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

Google didn't publish vit-tiny and vit-small model checkpoints in Hugging Face. I converted the weights from the [timm repository](https://github.com/rwightman/pytorch-image-models). This model is used in the same way as [ViT-base](https://huggingface.co/google/vit-base-patch16-224).
