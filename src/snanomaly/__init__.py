import sys

from loguru import logger

# Remove default handler
logger.remove()
# Add custom handler
logger.add(sys.stdout, format="{time} | {level} | {module}:{function}:{line} - {message}", level="INFO")
