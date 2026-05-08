import os
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigw,
    aws_apigatewayv2_integrations as apigw_integrations,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_ssm as ssm,
    CfnOutput,
)
from constructs import Construct

LAMBDA_DIR = os.path.join(os.path.dirname(__file__), "../../lambda")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../../frontend")


class InferenceStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, stage: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        is_dev = stage == "dev"

        # =====================================================================
        # 1) LAMBDA — thin proxy to SageMaker serverless endpoint
        # =====================================================================
        endpoint_name = f"hotdog-classifier-{stage}"

        fn = lambda_.DockerImageFunction(
            self, "PredictFn",
            function_name=f"hotdog-predict-{stage}",
            code=lambda_.DockerImageCode.from_image_asset(LAMBDA_DIR),
            memory_size=512,
            timeout=Duration.seconds(30),
            environment={
                "ENDPOINT_NAME": endpoint_name,
            },
        )

        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{endpoint_name}"],
        ))

        # =====================================================================
        # 3) HTTP API GATEWAY — CORS open for CloudFront origin
        # =====================================================================
        http_api = apigw.HttpApi(
            self, "PredictApi",
            api_name=f"hotdog-predict-{stage}",
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[apigw.CorsHttpMethod.POST],
                allow_headers=["Content-Type"],
            ),
        )

        http_api.add_routes(
            path="/predict",
            methods=[apigw.HttpMethod.POST],
            integration=apigw_integrations.HttpLambdaIntegration(
                "PredictIntegration", fn,
            ),
        )

        # =====================================================================
        # 4) FRONTEND S3 BUCKET — private, served only via CloudFront
        # =====================================================================
        frontend_bucket = s3.Bucket(
            self, "FrontendBucket",
            bucket_name=f"hotdog-frontend-{stage}-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY if is_dev else RemovalPolicy.RETAIN,
            auto_delete_objects=is_dev,
        )

        # =====================================================================
        # 5) CLOUDFRONT — OAC, HTTPS redirect, SPA 404→index.html
        # =====================================================================
        oac = cloudfront.S3OriginAccessControl(self, "OAC")

        distribution = cloudfront.Distribution(
            self, "FrontendCDN",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(
                    frontend_bucket,
                    origin_access_control=oac,
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                compress=True,
            ),
            default_root_object="index.html",
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                ),
            ],
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
        )

        # =====================================================================
        # 6) DEPLOY FRONTEND — static assets + CDK-generated config.json
        #    config.json is written at deploy time so app.js gets the live URL
        # =====================================================================
        s3deploy.BucketDeployment(
            self, "DeployFrontend",
            sources=[
                s3deploy.Source.asset(FRONTEND_DIR),
                s3deploy.Source.json_data("config.json", {
                    "apiUrl": http_api.api_endpoint,
                }),
            ],
            destination_bucket=frontend_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # =====================================================================
        # 7) SSM + OUTPUTS
        # =====================================================================
        ssm.StringParameter(
            self, "ApiUrlParam",
            parameter_name=f"/hotdog-mlops/{stage}/api-url",
            string_value=http_api.api_endpoint,
        )

        ssm.StringParameter(
            self, "FrontendUrlParam",
            parameter_name=f"/hotdog-mlops/{stage}/frontend-url",
            string_value=f"https://{distribution.distribution_domain_name}",
        )

        CfnOutput(
            self, "FrontendUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="Open this URL in your browser",
        )
        CfnOutput(
            self, "ApiEndpoint",
            value=http_api.api_endpoint,
            description="API Gateway — POST /predict",
        )
