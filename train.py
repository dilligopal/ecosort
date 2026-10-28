"""
EcoSort – Model Training Script
================================
Fine-tunes a pretrained MobileNetV2 on the Garbage Classification dataset
(12,000 images across 12 classes from a Kaggle-style directory structure)
and saves the weights as  models/ecosort_mobilenetv2.pth

Dataset used
------------
Garbage Classification – 12 categories:
  battery, biological, brown-glass, cardboard, clothes, green-glass,
  metal, paper, plastic, shoes, trash, white-glass

Mapping to EcoSort labels (8 training labels, 7 displayed)
-----------------------------------------------------------
  battery       → Electronic Waste
  biological    → Organic Waste
  brown-glass   → Glass
  cardboard     → Cardboard  (displayed as Paper)
  clothes       → Trash      (displayed as Unknown/Uncertain)
  green-glass   → Glass
  metal         → Metal
  paper         → Paper
  plastic       → Plastic
  shoes         → Trash      (displayed as Unknown/Uncertain)
  trash         → Trash      (displayed as Unknown/Uncertain)
  white-glass   → Glass

Because fine-grained glass colours merge into "Glass" and
clothes / shoes / generic trash merge into the "Trash" (uncertain) bucket,
we re-map before creating ImageFolder.

Usage
-----
1. Install dependencies:
       pip install -r requirements.txt

2. Download the dataset (one command, no Kaggle account needed):
       python train.py --download

   This downloads a public mirror of the garbage classification dataset
   (~220 MB) from Google Drive via gdown, extracts it, and creates a
   class-remapped dataset directory at ./dataset/

3. Train the model:
       python train.py --epochs 15 --batch-size 32

4. The trained model is saved to models/ecosort_mobilenetv2.pth

Notes
-----
- Training runs on CPU if no GPU is available (Render-compatible).
- GPU training is ~20× faster; use it locally if available.
- A ~15-epoch run on a modern CPU takes 30–90 min depending on hardware.
- The script performs a stratified 80/20 train/validation split.
- Validation accuracy is printed each epoch; best-validation weights are saved.
"""

import os
import sys
import shutil
import zipfile
import argparse
import logging
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
NUM_WORKERS = 0          # 0 = use main process (safe on Windows & CPU-only)
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ecosort_mobilenetv2.pth"
DATASET_DIR = Path("dataset")
RAW_DIR = Path("dataset_raw")

# Mapping from original Garbage Classification folder names → EcoSort training labels
FOLDER_TO_LABEL = {
    "battery":      "Electronic Waste",
    "biological":   "Organic Waste",
    "brown-glass":  "Glass",
    "cardboard":    "Cardboard",
    "clothes":      "Trash",
    "green-glass":  "Glass",
    "metal":        "Metal",
    "paper":        "Paper",
    "plastic":      "Plastic",
    "shoes":        "Trash",
    "trash":        "Trash",
    "white-glass":  "Glass",
}

# Final EcoSort training categories (alphabetical → consistent class index)
TRAINING_LABELS = sorted(set(FOLDER_TO_LABEL.values()))
# → ['Cardboard','Electronic Waste','Glass','Metal','Organic Waste','Paper','Plastic','Trash']

# These must match CATEGORY_LABELS in app.py (same alphabetical order)
assert TRAINING_LABELS == [
    "Cardboard", "Electronic Waste", "Glass", "Metal",
    "Organic Waste", "Paper", "Plastic", "Trash"
], "Label list mismatch – check FOLDER_TO_LABEL"

# ---------------------------------------------------------------------------
# Dataset Download & Preparation
# ---------------------------------------------------------------------------
GDRIVE_FILE_ID = "1BdHkEJFcDTsrSE_bfVlEYkYcz-JG4G5b"  # public mirror
# Alternative direct URL (no gdown needed):
FALLBACK_URL = (
    "https://github.com/garythung/trashnet/raw/master/"  # fallback not used; see below
)

KAGGLE_DOWNLOAD_CMD = (
    "kaggle datasets download -d asdasdasasdas/garbage-classification --unzip -p dataset_raw"
)

PUBLIC_MIRROR_URL = (
    "https://storage.googleapis.com/ecosort-assets/garbage_classification.zip"
)


def download_dataset():
    """
    Download and remap the Garbage Classification dataset.

    Tries (in order):
    1. gdown (Google Drive public mirror, no account required)
    2. Kaggle CLI (requires ~/.kaggle/kaggle.json)

    After download, images are copied into dataset/<label>/ directories
    so torchvision.datasets.ImageFolder can read them directly.
    """
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        log.info("Dataset directory '%s' already exists – skipping download.", DATASET_DIR)
        return

    zip_path = Path("garbage_classification.zip")

    # --- Attempt 1: gdown ---
    if not zip_path.exists():
        try:
            import gdown  # noqa: F401
            log.info("Downloading dataset via gdown …")
            gdown.download(id=GDRIVE_FILE_ID, output=str(zip_path), quiet=False)
        except ImportError:
            log.warning("gdown not installed (pip install gdown). Trying Kaggle CLI …")
        except Exception as exc:
            log.warning("gdown download failed: %s", exc)

    # --- Attempt 2: Kaggle CLI ---
    if not zip_path.exists():
        log.info("Trying Kaggle CLI: %s", KAGGLE_DOWNLOAD_CMD)
        ret = os.system(KAGGLE_DOWNLOAD_CMD)
        if ret != 0:
            log.error(
                "\n"
                "Could not download the dataset automatically.\n"
                "Manual download instructions:\n"
                "  1. Go to https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification\n"
                "  2. Download the ZIP file.\n"
                "  3. Place 'garbage_classification.zip' in this project directory.\n"
                "  4. Re-run: python train.py\n"
            )
            sys.exit(1)
        # Kaggle unzips directly; look for the folder
        kaggle_extracted = list(RAW_DIR.glob("Garbage classification"))
        if kaggle_extracted:
            _remap_dataset(kaggle_extracted[0])
            return

    # --- Extract ZIP ---
    if zip_path.exists():
        log.info("Extracting '%s' …", zip_path)
        RAW_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()  # free disk space

    # Find the root that contains the class folders
    raw_root = _find_dataset_root(RAW_DIR)
    _remap_dataset(raw_root)


def _find_dataset_root(base: Path) -> Path:
    """Walk base until we find a directory that contains class sub-folders."""
    known_folders = set(FOLDER_TO_LABEL.keys())
    for root, dirs, _ in os.walk(base):
        if set(dirs) & known_folders:
            return Path(root)
    raise RuntimeError(
        f"Could not locate class folders inside '{base}'. "
        "Check the extracted archive structure."
    )


def _remap_dataset(raw_root: Path):
    """
    Copy images from raw_root/<original_label>/... into dataset/<mapped_label>/...
    merging green-glass / brown-glass / white-glass → Glass, etc.
    """
    log.info("Remapping dataset from '%s' → '%s' …", raw_root, DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Create output class directories
    for label in TRAINING_LABELS:
        (DATASET_DIR / label).mkdir(exist_ok=True)

    total_copied = 0
    for folder in raw_root.iterdir():
        if not folder.is_dir():
            continue
        mapped_label = FOLDER_TO_LABEL.get(folder.name.lower())
        if mapped_label is None:
            log.warning("Unknown folder '%s' – skipping.", folder.name)
            continue
        dest_dir = DATASET_DIR / mapped_label
        for img_path in folder.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                dest_file = dest_dir / img_path.name
                # Avoid name collisions when merging glass sub-types
                if dest_file.exists():
                    stem = img_path.stem + f"_{folder.name}"
                    dest_file = dest_dir / (stem + img_path.suffix)
                shutil.copy2(img_path, dest_file)
                total_copied += 1

    log.info("Remapping complete: %d images copied.", total_copied)

    # Class distribution summary
    for label in TRAINING_LABELS:
        count = len(list((DATASET_DIR / label).iterdir()))
        log.info("  %-20s: %d images", label, count)


# ---------------------------------------------------------------------------
# Data Augmentation & Preprocessing
# ---------------------------------------------------------------------------
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    MobileNetV2 with pretrained ImageNet weights + custom classification head.
    Phase 1: freeze all feature layers → only train classifier.
    Phase 2: unfreeze all → fine-tune end-to-end with lower LR.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def unfreeze_backbone(model: nn.Module):
    """Unfreeze feature extractor for fine-tuning phase."""
    for param in model.features.parameters():
        param.requires_grad = True
    log.info("Backbone unfrozen – full network will be fine-tuned.")


def stratified_split(dataset, val_ratio=0.20):
    """Return Subset objects for train and validation using stratified split."""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(sss.split(targets, targets))
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            batch_acc = correct / total * 100
            log.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f | Acc: %.1f%%",
                epoch, batch_idx + 1, len(loader), running_loss / total, batch_acc,
            )

    return running_loss / total, correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total * 100


def train(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

    if not DATASET_DIR.exists():
        log.error(
            "Dataset not found at '%s'. Run:  python train.py --download", DATASET_DIR
        )
        sys.exit(1)

    # Build dataset with train transforms; we'll swap for val subset manually
    full_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=TRAIN_TRANSFORMS)
    num_classes = len(full_dataset.classes)
    log.info("Classes (%d): %s", num_classes, full_dataset.classes)

    # Verify labels match expected order
    assert full_dataset.classes == TRAINING_LABELS, (
        f"Class mismatch!\nExpected: {TRAINING_LABELS}\nFound: {full_dataset.classes}\n"
        "Ensure dataset directory uses the correct label folder names."
    )

    train_subset, val_subset = stratified_split(full_dataset, val_ratio=0.20)

    # Apply validation transforms to val subset via a wrapper
    val_dataset = _TransformSubset(full_dataset, val_subset.indices, VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    log.info("Train: %d samples | Val: %d samples", len(train_subset), len(val_dataset))

    # Model
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -----------------------------------------------------------------------
    # Phase 1: Train classifier head only (frozen backbone)
    # -----------------------------------------------------------------------
    PHASE1_EPOCHS = max(1, args.epochs // 3)
    PHASE2_EPOCHS = args.epochs - PHASE1_EPOCHS

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    MODEL_DIR.mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("PHASE 1 – Classifier warm-up (%d epochs)", PHASE1_EPOCHS)
    log.info("=" * 60)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "Epoch %2d/%d | Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | %.0fs",
            epoch, PHASE1_EPOCHS, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.info("  ✓ Best model saved (val_acc=%.1f%%)", best_val_acc)

    # -----------------------------------------------------------------------
    # Phase 2: Full fine-tuning (unfreeze backbone)
    # -----------------------------------------------------------------------
    if PHASE2_EPOCHS > 0:
        unfreeze_backbone(model)

        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

        log.info("=" * 60)
        log.info("PHASE 2 – Full fine-tuning (%d epochs)", PHASE2_EPOCHS)
        log.info("=" * 60)

        for epoch in range(1, PHASE2_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch + PHASE1_EPOCHS
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            log.info(
                "Epoch %2d/%d | Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | %.0fs",
                epoch, PHASE2_EPOCHS, train_loss, train_acc, val_loss, val_acc, elapsed,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                log.info("  ✓ Best model saved (val_acc=%.1f%%)", best_val_acc)

    log.info("Training complete. Best validation accuracy: %.1f%%", best_val_acc)
    log.info("Model saved to: %s", MODEL_PATH)
    log.info("Now start the Flask server:  python app.py")


# ---------------------------------------------------------------------------
# Helper: Apply a different transform to a Subset without cloning data
# ---------------------------------------------------------------------------
class _TransformSubset(torch.utils.data.Dataset):
    """Wraps an ImageFolder subset and applies a different transform."""

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        from PIL import Image as _PIL
        img = _PIL.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EcoSort MobileNetV2 Training Script"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download and prepare the garbage classification dataset before training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Total training epochs (default: 15). Phase 1 = epochs//3, Phase 2 = rest.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32). Reduce to 16 if you run out of memory.",
    )
    args = parser.parse_args()

    if args.download:
        download_dataset()

    train(args)

