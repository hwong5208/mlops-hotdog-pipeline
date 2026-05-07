import sys
import boto3

REGION = "us-west-2"
STAGE = "dev"


def get_ssm(name):
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]


def find_latest_completed_job(sm):
    resp = sm.list_training_jobs(
        NameContains=f"hotdog-train-{STAGE}-",
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    jobs = resp.get("TrainingJobSummaries", [])
    if not jobs:
        raise SystemExit(f"No completed training jobs found for stage '{STAGE}'")
    return jobs[0]["TrainingJobName"]


def main():
    job_name = sys.argv[1] if len(sys.argv) > 1 else None

    sm = boto3.client("sagemaker", region_name=REGION)
    pkg_group = get_ssm(f"/hotdog-mlops/{STAGE}/model-package-group-name")

    if job_name is None:
        job_name = find_latest_completed_job(sm)
        print(f"Using latest completed job: {job_name}")

    job = sm.describe_training_job(TrainingJobName=job_name)
    model_uri = job["ModelArtifacts"]["S3ModelArtifacts"]

    metrics = {m["MetricName"]: m["Value"] for m in job.get("FinalMetricDataList", [])}
    test_acc = metrics.get("test_acc")
    test_loss = metrics.get("test_loss")

    print(f"Job:       {job_name}")
    print(f"Model:     {model_uri}")
    if test_acc is not None:
        print(f"test_acc:  {test_acc:.4f}  |  test_loss: {test_loss:.4f}")
    else:
        print("  [warn] No metrics found — job may not have used MetricDefinitions")

    description = f"job={job_name}"
    if test_acc is not None:
        description += f" test_acc={test_acc:.4f}"

    metadata = {
        "training_job": job_name,
        "model_s3_uri": model_uri,
    }
    if test_acc is not None:
        metadata["test_acc"] = f"{test_acc:.4f}"
    if test_loss is not None:
        metadata["test_loss"] = f"{test_loss:.4f}"

    resp = sm.create_model_package(
        ModelPackageGroupName=pkg_group,
        ModelApprovalStatus="PendingManualApproval",
        ModelPackageDescription=description,
        CustomerMetadataProperties=metadata,
    )
    pkg_arn = resp["ModelPackageArn"]

    print(f"\nRegistered  → {pkg_arn}")
    print("Status       : PendingManualApproval")
    print("\nApprove with:")
    print(f'  aws sagemaker update-model-package \\')
    print(f'    --model-package-arn "{pkg_arn}" \\')
    print(f'    --model-approval-status Approved')


if __name__ == "__main__":
    main()
