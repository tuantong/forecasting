import urllib.request
from io import BytesIO

import pandas as pd
from PIL import Image


def download_and_read_image(url):
    """
    Downloads an image from a URL and returns it as a PIL Image object.
    If the download fails or the URL is None, returns None.

    Parameters:
    - url: URL of the image to download.

    Returns:
    - The downloaded Image object or None if the download fails.
    """
    if url is None or pd.isna(url) or url.lower() in ["na", "<na>", "null", "none"]:
        #         print("No URL provided, skipping download.")
        return None
    try:
        with urllib.request.urlopen(url) as res:
            image = Image.open(BytesIO(res.read()))

            image = image.convert("RGB")
            # TODO:
            # Convert images to RGB, dropping alpha if present
        #             if image.mode in ["RGBA", "LA"] or (
        #                 image.mode == "P" and "transparency" in image.info
        #             ):
        #                 image = image.convert("RGBA")  # Ensure it's RGBA to preserve blending
        #                 background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        #                 background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        #                 image = background.convert("RGB")
        #             else:
        #                 image = image.convert("RGB")

        return image
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None
