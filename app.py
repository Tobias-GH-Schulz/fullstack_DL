import logging
import requests

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/submit", methods=["GET"])
def home():

    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def post_image():

    
    if request.method == "POST":
        # we will get the file from the request
        file = request.files["file"]
        # convert that to bytes
        img_bytes = file.read()

    file = request.files["file"]

    r = requests.post("....", files={"file": file})

    r.raise_for_status()
