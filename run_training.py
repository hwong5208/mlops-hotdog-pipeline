import time
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
STAGE = "dev"


def get_ssm(name):
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]


def preflight(bucket, ecr_uri):
    """Fail fast locally before spending money on a SageMaker job."""
    s3 = boto3.client("s3", region_name=REGION)
    ecr = boto3.client("ecr", region_name=REGION)

    # 1. ECR image must exist
    repo_name = ecr_uri.split("/")[-1].split(":")[0]
    try:
        ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": "latest"}])
        print("  [ok] ECR image exists")
    except ecr.exceptions.ImageNotFoundException:
        raise SystemExit(f"  [fail] ECR image not found: {ecr_uri} — rebuild and push first")

    # 2. Training data must exist in S3
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="data/", MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        raise SystemExit(f"  [fail] No training data at s3://{bucket}/data/ — upload data first")
    print("  [ok] Training data exists in S3")

    print("Pre-flight passed.\n")


def main():
    bucket = get_ssm(f"/hotdog-mlops/{STAGE}/bucket-name")
    ecr_uri = get_ssm(f"/hotdog-mlops/{STAGE}/ecr-repo-uri")
    role_arn = get_ssm(f"/hotdog-mlops/{STAGE}/sagemaker-role-arn")

    print(f"Bucket:  {bucket}")
    print(f"ECR URI: {ecr_uri}")
    print(f"Role:    {role_arn}")

    print("\nRunning pre-flight checks...")
    preflight(bucket, ecr_uri)

    job_name = f"hotdog-train-{STAGE}-{int(time.time())}"
    sm = boto3.client("sagemaker", region_name=REGION)

    try:
        sm.create_experiment(ExperimentName="hotdog-classifier")
        print("Created SageMaker experiment: hotdog-classifier")
    except ClientError:
        pass  # already exists

    sm.create_trial(ExperimentName="hotdog-classifier", TrialName=job_name)

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": f"{ecr_uri}:latest",
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "train_loss", "Regex": "train_loss=([0-9\\.]+)"},
                {"Name": "train_acc",  "Regex": "train_acc=([0-9\\.]+)"},
                {"Name": "test_loss",  "Regex": "test_loss=([0-9\\.]+)"},
                {"Name": "test_acc",   "Regex": "test_acc=([0-9\\.]+)"},
            ],
        },
        ExperimentConfig={
            "ExperimentName": "hotdog-classifier",
            "TrialName": job_name,
            "TrialComponentDisplayName": job_name,
        },
        RoleArn=role_arn,
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }],
        OutputDataConfig={
            "S3OutputPath": f"s3://{bucket}/output/",
        },
        ResourceConfig={
            "InstanceType": "ml.g4dn.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 1800,
            "MaxWaitTimeInSeconds": 3600,
        },
        EnableManagedSpotTraining=True,
        Environment={"GIT_PYTHON_REFRESH": "quiet"},
    )

    print(f"\nJob submitted: {job_name}")
    print("Waiting for training to complete...")

    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        print(f"  Status: {status} | {secondary}")

        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(30)

    if status == "Completed":
        model_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"\nDone! Model artifact: {model_uri}")
    else:
        reason = resp.get("FailureReason", "unknown")
        print(f"\nJob {status}: {reason}")


if __name__ == "__main__":
    main()
