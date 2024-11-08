import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from .yolov9 import YOLOv9

__all__ = ["YOLOv9", "ROOT"]  # allow simpler import
