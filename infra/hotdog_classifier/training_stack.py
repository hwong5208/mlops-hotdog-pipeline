import json
import os
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_sagemaker as sagemaker,
    CfnOutput,
)
from constructs import Construct


class TrainingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, stage: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        config_path = os.path.join(os.path.dirname(__file__), "../config.json")
        with open(config_path) as f:
            config = json.load(f)[stage]

        is_dev = stage == "dev"

        # ================================================================
        # 1) S3 BUCKET — training data input + model artifact output
        # ================================================================
        self.bucket = s3.Bucket(
            self, "TrainingBucket",
            bucket_name=f"{config['bucket_name']}-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY if is_dev else RemovalPolicy.RETAIN,
            auto_delete_objects=is_dev,
        )

        # ================================================================
        # 2) ECR REPOSITORY — Docker image for SageMaker training job
        # ================================================================
        self.ecr_repo = ecr.Repository(
            self, "TrainingRepo",
            repository_name=config["ecr_repo_name"],
            removal_policy=RemovalPolicy.DESTROY if is_dev else RemovalPolicy.RETAIN,
            empty_on_delete=is_dev,
        )

        # ================================================================
        # 3) SAGEMAKER EXECUTION ROLE
        # ================================================================
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            role_name=f"hotdog-sagemaker-role-{stage}",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )

        self.bucket.grant_read_write(self.sagemaker_role)
        self.ecr_repo.grant_pull(self.sagemaker_role)

        self.sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            resources=["*"],
        ))

        # ================================================================
        # 4) MODEL REGISTRY — package group for versioned model approvals
        # ================================================================
        model_pkg_group_name = f"hotdog-classifier-{stage}"

        sagemaker.CfnModelPackageGroup(
            self, "ModelPackageGroup",
            model_package_group_name=model_pkg_group_name,
            model_package_group_description=f"Hotdog classifier models — {stage}",
        )

        # ================================================================
        # 5) SSM PARAMETERS — consumed by run_training.py script
        # ================================================================
        ssm.StringParameter(
            self, "BucketNameParam",
            parameter_name=f"/hotdog-mlops/{stage}/bucket-name",
            string_value=self.bucket.bucket_name,
        )

        ssm.StringParameter(
            self, "EcrRepoUriParam",
            parameter_name=f"/hotdog-mlops/{stage}/ecr-repo-uri",
            string_value=self.ecr_repo.repository_uri,
        )

        ssm.StringParameter(
            self, "SageMakerRoleArnParam",
            parameter_name=f"/hotdog-mlops/{stage}/sagemaker-role-arn",
            string_value=self.sagemaker_role.role_arn,
        )

        ssm.StringParameter(
            self, "ModelPackageGroupParam",
            parameter_name=f"/hotdog-mlops/{stage}/model-package-group-name",
            string_value=model_pkg_group_name,
        )

        # ================================================================
        # 6) GITHUB ACTIONS OIDC — keyless auth for CI/CD deploy
        # ================================================================
        github_oidc = iam.OpenIdConnectProvider(
            self, "GitHubOIDC",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )

        github_deploy_role = iam.Role(
            self, "GitHubDeployRole",
            role_name=f"hotdog-github-deploy-{stage}",
            assumed_by=iam.WebIdentityPrincipal(
                github_oidc.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub":
                            "repo:hwong5208/mlops-hotdog-pipeline:*",
                    },
                },
            ),
        )

        github_deploy_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeModel",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:ListModelPackages",
                "sagemaker:DescribeModelPackage",
            ],
            resources=["*"],
        ))

        github_deploy_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[self.sagemaker_role.role_arn],
        ))

        github_deploy_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "ecr:GetAuthorizationToken",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
            ],
            resources=["*"],
        ))

        github_deploy_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[f"{self.bucket.bucket_arn}/*"],
        ))

        github_deploy_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/hotdog-mlops/{stage}/*"],
        ))

        ssm.StringParameter(
            self, "GitHubDeployRoleParam",
            parameter_name=f"/hotdog-mlops/{stage}/github-deploy-role-arn",
            string_value=github_deploy_role.role_arn,
        )

        # ================================================================
        # 7) OUTPUTS
        # ================================================================
        CfnOutput(self, "BucketName", value=self.bucket.bucket_name, description="S3 bucket for data and artifacts")
        CfnOutput(self, "EcrRepoUri", value=self.ecr_repo.repository_uri, description="ECR repo URI for training image")
        CfnOutput(self, "SageMakerRoleArn", value=self.sagemaker_role.role_arn, description="SageMaker execution role ARN")
        CfnOutput(self, "ModelPackageGroupName", value=model_pkg_group_name, description="SageMaker Model Registry group")
        CfnOutput(self, "GitHubDeployRoleArn", value=github_deploy_role.role_arn, description="IAM role for GitHub Actions deployment")
