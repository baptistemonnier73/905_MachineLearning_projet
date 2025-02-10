import io
import json

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image


def display_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image_cv)
    cv2.waitKey(500)

name = "Adèle Exarchopoulos"

dataset_url = "https://huggingface.co/datasets/tonyassi/celebrity-1000/resolve/main/data/train-00000-of-00001.parquet"

dataset_get_response = requests.get(dataset_url)

if dataset_get_response.status_code == 200:

    with open("../data/names.json", "r", encoding="utf-8") as f:

        names = json.load(f)

        dataset = pd.read_parquet(io.BytesIO(dataset_get_response.content), engine="pyarrow")

        label = int(next(iter({name_label for name_label, v in names.items() if v == name}), None))

        images_bytes = {dataset["image"][image_index]["bytes"] for image_index, dataset_label in dataset["label"].items() if dataset_label == label}

        print(f"name: {name}, label:{label}, {len(images_bytes)} images")

        for image_bytes in images_bytes:
            display_image(image_bytes)

    cv2.destroyAllWindows()

else:
    print("error getting dataset")