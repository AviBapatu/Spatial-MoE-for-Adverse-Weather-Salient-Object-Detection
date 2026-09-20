import os
from src.dataset import get_weather_type

def parse_weather_from_path(filepath: str) -> str:
    """Extract weather condition from a full filepath (thin wrapper)."""
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0]
    return get_weather_type(stem)
