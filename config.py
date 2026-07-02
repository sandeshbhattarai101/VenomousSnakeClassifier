from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = MODELS_DIR / "snake_classifier.pth"
CONFIG_PATH = MODELS_DIR / "model_config.json"
PLOTS_DIR = PROJECT_ROOT / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 20
PHASE1_LR = 3e-4
PHASE2_LR = 5e-6
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE = 6

MODEL_NAME = "efficientnet_b2"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# Test = remaining 0.10

CLASSES = ["non_venomous", "venomous"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
