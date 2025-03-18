import sys
from pathlib import Path

from loguru import logger

### Configure logger ###

# Remove default handler
logger.remove()
# Add custom handler
logger.add(sys.stdout, format="{time} | {level} | {module}:{function}:{line} - {message}", level="INFO")


### Predefined paths ###

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DATASETS_DIR = PROJECT_DIR / "datasets"
