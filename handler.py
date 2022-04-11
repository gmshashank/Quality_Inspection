try:
    import unzip_requirements
except ImportError:
    pass

import base64
from json import decoder
import json
import boto3
import io
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from requests_toolbelt.multipart import decoder

# define env variables if they do not exist
S3_BUCKET = (
    os.environ["S3_BUCKET"] if "S3_BUCKET" is os.environ else "manufacturing-models"
)
MODEL_PATH = (
    os.environ["MODEL_PATH"] if "MODEL_PATH" in os.environ else "./mobilenet_v2.pt"
)

print(f"Downloading model: {MODEL_PATH}")

# Loading S3 client when Lambda execution context is created
s3 = boto3.client("s3")


def load_model_from_s3():
    """
    Load model from S3 bucket
    """
    try:
        if os.path.isfile(MODEL_PATH) != True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
            print("Creating bytestream")
            bytestream = io.BytesIO(obj["Body"].read())
            print("Loading model")
            model = torch.jit.load(bytestream)
            print("Model loaded successfully.")
            return model
        else:
            print("Model loading failed from path: {MODEL_PATH}.")
    except Exception as e:
        print(repr(e))
        raise (e)


model = load_model_from_s3()


def transform_image(image_bytes):
    """
    Transform the image
    """
    try:
        transformations = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise (e)


def get_prediction(image_bytes):
    """
    transform image and predict the model output
    """
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def classify_image(event, context):
    """
    Load image and predict output
    """
    try:
        content_type_header = event["headers"]["content-type"]
        print("Image content: " + event["body"])
        body = base64.b64decode(event["body"])
        print("Body Loaded")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print("Image classification code: ", prediction)

        filename = (
            picture.headers[b"Content-Disposition"].decode().split(";")[1].split("=")[1]
        )
        if len(filename) < 4:
            filename = (
                picture.headers[b"Content-Disposition"]
                .decode()
                .split(";")[2]
                .split("=")[1]
            )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps(
                {"file": filename.replace('"', ""), "predicted": prediction}
            ),
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }
