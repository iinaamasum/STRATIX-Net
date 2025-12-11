"""
Configuration and constants for STRATIX-Net models
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Class name aliases for confusion matrix
CLASS_ALIASES = {
    "normal-cecum": "Cecum",
    "dyed-lifted-polyps": "DLP",
    "dyed-resection-margins": "DRM",
    "esophagitis": "Esophagitis",
    "normal-pylorus": "Pylorus",
    "polyps": "Polyps",
    "ulcerative-colitis": "UC",
    "normal-z-line": "Z-Line",
}

# Expected classes
EXPECTED_CLASSES = [
    "normal-cecum",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-pylorus",
    "polyps",
    "ulcerative-colitis",
    "normal-z-line",
]
