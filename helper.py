import io
import base64
import requests
from PIL import Image


max_pixels = 1568 * 1568
def resize_image(img: Image.Image) -> str:
    w, h = img.size
    if w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        img.thumbnail((int(w*ratio), int(h*ratio)))
    return img

def base64_from_image(src: str) -> str:
    resp = requests.get(src)
    resp.raise_for_status()
    img_bytes = resp.content

    img = Image.open(io.BytesIO(img_bytes))
    orig_format = img.format or 'PNG'

    img = resize_image(img)

    with io.BytesIO() as img_buffer:
        img.save(img_buffer, format=orig_format)
        img_buffer.seek(0)
        encoded = base64.b64encode(img_buffer.read()).decode('utf-8')
        return f"data:image/{orig_format.lower()};base64,{encoded}"



def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url)
    resp.raise_for_status()              
    return Image.open(io.BytesIO(resp.content))