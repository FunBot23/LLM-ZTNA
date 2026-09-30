"""Reset the resource CSVs to their seed state.

Run between demos so the sensor data, device config, and audit log all
start from a known state.

Usage from project root:
    python scripts/reset_resources.py
"""
import sys
from pathlib import Path

# Make the app package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import resources

resources.reset()
print(f"reset resource files under {resources.RESOURCES_DIR}")
