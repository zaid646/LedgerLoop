# Model Card — ViT Receipt GL-Classification

Vision Transformer fine-tuned for LedgerLoop receipt general-ledger (GL)
category classification. Classifies the full receipt image into one of 13 GL
category codes.

## Quick facts

| Property | Value |
|---|---|
| Task | Multi-class image classification (13 GL categories) |
| Base model | `google/vit-base-patch16-224` |
| Fine-tune data | Synthetic receipt dataset (cv/data/): 150 train / 50 val images |
| Eval data | 50 held-out images (25 unseen docs × 2 variants) |
| Framework | PyTorch + HuggingFace Transformers + torchvision |
| Input resolution | 224×224 |

## GL categories (13)

TRAVEL, MEALS, OFFICE_SUPPLIES, SOFTWARE, HARDWARE, MARKETING,
PROFESSIONAL_SERVICES, UTILITIES, RENT, INSURANCE, TRAINING,
ENTERTAINMENT, OTHER.

## Dataset layout

The ViT expects images sorted into GL category folders::

    cv/data/vit_train/
        TRAVEL/   *.jpg
        MEALS/    *.jpg
        ...
        OTHER/    *.jpg
    cv/data/vit_val/
        TRAVEL/   *.jpg
        ...

Folder names must match GL_CATEGORIES exactly. The synthetic receipt dataset is
generated locally under `cv/data/` (gitignored) and contains images and
YOLO-style labels; the ViT dataset is derived by grouping images by their
dominant GL category.

## Training

See [`cv/train_vit.py`](cv/train_vit.py). Defaults: 10 epochs, batch 16,
lr 2e-5, linear warmup (3%), AdamW, image size 224. Trains on CPU or CUDA.

## Results

Evaluation on 50 held-out receipt images:

| Model | Accuracy | Macro F1 | Train runtime | Device |
|---|---:|---:|---:|---:|
| ViT base (seed 42) | TBD | TBD | TBD | TBD |

Fill in TBD after the remote GPU training run completes.

## Comparison with YOLOv8n (object detection baseline)

YOLOv8n detects receipt fields (merchant, date, total) as bounding boxes:

| Model | Task | Key metric |
|---|---|---|
| YOLOv8n | Field detection (bbox) | mAP50 = 0.995, mAP50-95 = 0.741 |
| ViT base | GL classification (image) | accuracy / macro F1 (TBD) |

Different tasks — ViT classifies the whole receipt into a GL category;
YOLOv8n localizes individual fields. Both measure receipt understanding.

## Inference

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

model = AutoModelForImageClassification.from_pretrained("./cv/outputs/vit-base/final")
processor = AutoImageProcessor.from_pretrained("./cv/outputs/vit-base/final")

image = Image.open("receipt.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits
pred = logits.argmax(dim=-1).item()
print(model.config.id2label[pred])  # → SOFTWARE
```

## Files

```
cv/
├── train_vit.py     # ViT fine-tuning script (CPU + CUDA)
├── compare_vit_yolo.py  # ViT vs YOLOv8n comparison
├── MODEL_CARD.md    # this file
├── outputs/         # fine-tuned model checkpoints (gitignored)
├── results/         # eval JSON + metrics (gitignored)
├── data/
│   ├── vit_train/   # training images by GL category (to be prepared)
│   └── vit_val/     # validation images by GL category (to be prepared)
└── tests/           # unit tests
```

## Citation

If you use this model card or training script, please reference the LedgerLoop
repository.
