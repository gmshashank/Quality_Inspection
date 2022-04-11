import torch
import boto3


def upload_model(
    model_path: str = "",
    s3_bucket: str = "",
    key_prefix: str = "",
    aws_profile: str = "default",
):
    """
    Upload model to S3 bucket
    """
    s3 = boto3.session.Session(profile_name=aws_profile)
    client = s3.client("s3")
    client.upload_file(model_path, s3_bucket, key_prefix)


if __name__ == "__main__":
    S3_BUCKET = "manufacturing-models"
    KEY_PREFIX = "mobilenet_v2.pt"
    SRC_MODEL_PATH = "../mobilenet_v2.pt"

    upload_model(SRC_MODEL_PATH, S3_BUCKET, KEY_PREFIX)
    print(f"{SRC_MODEL_PATH} is uploaded to S3 Bucket --> {S3_BUCKET}/{KEY_PREFIX}")
