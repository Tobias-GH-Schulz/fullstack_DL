import io
import sys
import json
from functools import lru_cache

from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
import torchvision
import torch

import logging

from flask import Flask, jsonify, request

MODELS_DIR = Path("../models")
ASSETS_DIR = Path("../assets")
MODEL_NAME = "best_model.pth"
FIRST_STAGE_NAME = "first_stage_model.pth"

log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))

LOG_LEVEL = "debug"

if LOG_LEVEL == "info":
    log.setLevel(logging.INFO)
else:
    log.setLevel(logging.DEBUG)

@lru_cache()
def load_model():
    model = torchvision.models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class2idx = json.loads(
        Path(ASSETS_DIR / "class2idx.json").read_text()
    )
    idx2class = {v: k for k, v in class2idx.items()}

    model.load_state_dict(
        torch.load(MODELS_DIR / MODEL_NAME)
    )

    model = model.eval()

    return model, idx2class

@lru_cache()
def load_first_model():
    model = torchvision.models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class2idx = json.loads(
        Path(ASSETS_DIR / "first_stage_class2idx.json").read_text()
    )
    idx2class = {v: k for k, v in class2idx.items()}

    model.load_state_dict(
        torch.load(MODELS_DIR / FIRST_STAGE_NAME)
    )

    model = model.eval()

    return model, idx2class


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def main_function():
    if request.method == "POST":
        # we will get the file from the request
        file = request.files["file"]
        # convert that to bytes
        img_bytes = file.read()
    
    first_class = first_stage(img_bytes)

    if first_class == "ants_bees":
        ant_or_bee = predict(img_bytes)
        return ant_or_bee
    else:
        return jsonify("This is not an ant or a bee. Please upload another image.")


def first_stage(img_bytes):
    pred_class = get_prediction(image_bytes=img_bytes, models="first_stage")
    return pred_class
    # return jsonify({"class_id": class_id, "class_name": class_name})


def predict(img_bytes):
    pred_class = get_prediction(image_bytes=img_bytes, models="second_stage")
    return jsonify(pred_class)
    # return jsonify({"class_id": class_id, "class_name": class_name})


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# classifier_idx_data = json.load(open("whatever.json"))

def get_prediction(image_bytes, models):
    if models == "first_stage":
        model, idx2class = load_first_model()
    else:
        model, idx2class = load_model()
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return idx2class[int(predicted_idx)]
    



if __name__ == "__main__":
    app.run()


# run with
# FLASK_ENV=development FLASK_APP=app.py flask run

# import requests

# resp = requests.post("http://localhost:5000/predict",
#                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})
