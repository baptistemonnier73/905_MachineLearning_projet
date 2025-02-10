import io
import json

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # Utiliser un format explicite
    return buffer.getvalue()

def display_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image_cv)
    cv2.waitKey(500)

name = "Barack Obama"


with open("../data/names.json", "r", encoding="utf-8") as f:

    names = json.load(f)

    dataset = load_dataset("tonyassi/celebrity-1000")["train"]

    label = int(next(iter({name_label for name_label, v in names.items() if v == name}), None))

    images_bytes = {image_to_bytes(image_dataset) for image_dataset, label_dataset in zip(dataset["image"], dataset["label"]) if label_dataset == label}

    print(f"name: {name}, label:{label}, {len(images_bytes)} images")

    for image_bytes in images_bytes:
        display_image(image_bytes)

    cv2.destroyAllWindows()
