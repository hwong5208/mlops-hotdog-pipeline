import sys
import io
import tarfile
import tempfile
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

REGION = "us-west-2"
STAGE = "dev"
ENDPOINT_NAME = f"hotdog-classifier-{STAGE}"
# SageMaker PyTorch CPU inference container (us-west-2)
INFERENCE_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"


def get_ssm(name):
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]


def get_approved_model(sm, pkg_group, pkg_arn=None):
    if pkg_arn:
        return sm.describe_model_package(ModelPackageName=pkg_arn)
    resp = sm.list_model_packages(
        ModelPackageGroupName=pkg_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        raise SystemExit("No approved model packages found — approve a model first.")
    return sm.describe_model_package(ModelPackageName=pkgs[0]["ModelPackageArn"])


def repackage_model(s3, model_uri, bucket, inference_py_path):
    """Add code/inference.py into model.tar.gz so the PyTorch DLC can serve it."""
    s3_key = model_uri.replace(f"s3://{bucket}/", "")
    inference_key = s3_key.replace("output/model.tar.gz", "inference/model.tar.gz")

    with tempfile.TemporaryDirectory() as tmp:
        local_tar = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, s3_key, str(local_tar))

        out_buf = io.BytesIO()
        with tarfile.open(local_tar, "r:gz") as src, \
             tarfile.open(fileobj=out_buf, mode="w:gz") as dst:
            for member in src.getmembers():
                dst.addfile(member, src.extractfile(member))
            data = inference_py_path.read_bytes()
            info = tarfile.TarInfo(name="code/inference.py")
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))

        out_buf.seek(0)
        s3.upload_fileobj(out_buf, bucket, inference_key)

    inference_uri = f"s3://{bucket}/{inference_key}"
    print(f"  [ok] Repackaged model → {inference_uri}")
    return inference_uri


def deploy(sm, model_uri, role_arn, suffix):
    model_name = f"{ENDPOINT_NAME}-{suffix}"
    config_name = model_name

    try:
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": INFERENCE_IMAGE,
                "ModelDataUrl": model_uri,
                "Environment": {"SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"},
            },
        )
        print(f"  [ok] Created model: {model_name}")
    except ClientError as e:
        if "already exists" in str(e):
            print(f"  [ok] Model already exists: {model_name}")
        else:
            raise

    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "primary",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": 2048,
                    "MaxConcurrency": 5,
                },
            }],
        )
        print(f"  [ok] Created endpoint config: {config_name}")
    except ClientError as e:
        if "already exists" in str(e):
            print(f"  [ok] Endpoint config already exists: {config_name}")
        else:
            raise

    try:
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        print(f"  [ok] Creating endpoint: {ENDPOINT_NAME}")
    except ClientError as e:
        if "already exists" in str(e):
            sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
            print(f"  [ok] Updating endpoint: {ENDPOINT_NAME}")
        else:
            raise


def main():
    pkg_arn = sys.argv[1] if len(sys.argv) > 1 else None

    inference_py = Path(__file__).parent / "inference.py"
    if not inference_py.exists():
        raise SystemExit(f"inference.py not found at {inference_py}")

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    bucket = get_ssm(f"/hotdog-mlops/{STAGE}/bucket-name")
    role_arn = get_ssm(f"/hotdog-mlops/{STAGE}/sagemaker-role-arn")
    pkg_group = get_ssm(f"/hotdog-mlops/{STAGE}/model-package-group-name")

    pkg = get_approved_model(sm, pkg_group, pkg_arn)
    metadata = pkg["CustomerMetadataProperties"]
    job_name = metadata["training_job"]
    model_s3_uri = metadata["model_s3_uri"]
    test_acc = metadata.get("test_acc", "?")

    print(f"Job:      {job_name}")
    print(f"test_acc: {test_acc}")
    print(f"Artifact: {model_s3_uri}\n")

    inference_uri = repackage_model(s3, model_s3_uri, bucket, inference_py)
    deploy(sm, inference_uri, role_arn, job_name[-10:])

    print(f"\nEndpoint '{ENDPOINT_NAME}' deploying — takes ~5 min to become InService.")
    print(f"\nCheck status:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION} --query EndpointStatus")


if __name__ == "__main__":
    main()
