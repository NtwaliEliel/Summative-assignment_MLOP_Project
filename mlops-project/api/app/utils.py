"""Utility functions for the Iris Classification API."""

import logging
from typing import Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory (mlops-project)."""
    current = Path(__file__).resolve()
    # Navigate from api/app/utils.py to mlops-project root
    return current.parent.parent.parent


def format_response(status: str, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format a standardized API response."""
    response = {
        "status": status,
        "message": message
    }
    if data is not None:
        response["data"] = data
    return response


# Label mapping for Iris species
LABEL_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

LABEL_NAMES = ["setosa", "versicolor", "virginica"]

