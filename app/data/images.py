from typing import Any
from datetime import datetime
import requests

_IMAGE_URL = "http://jsoc.stanford.edu/doc/data/hmi/harp/harp_definitive/%Y/%m/%d/harp.%Y.%m.%d_%H:00:00_TAI.png"


def plot_sharp_image(placeholder: Any, end: datetime):
    url = datetime.fromtimestamp(end).strftime(_IMAGE_URL)
    response = requests.get(url)
    if response.status_code == 200:
        placeholder.image(url, width=300)
        return True
    else:
        return False
