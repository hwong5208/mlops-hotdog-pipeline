import os
import json
import base64
import boto3

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

sm_runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-west-2"))


def handler(event, context):
    try:
        body = event.get("body", "")
        if event.get("isBase64Encoded"):
            image_bytes = base64.b64decode(body)
        else:
            data = json.loads(body)
            image_bytes = base64.b64decode(data["image"])

        resp = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="image/jpeg",
            Body=image_bytes,
        )
        result = json.loads(resp["Body"].read())

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "prediction": result["class"],
                "confidence": result["confidence"],
            }),
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
