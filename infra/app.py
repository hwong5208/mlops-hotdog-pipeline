#!/usr/bin/env python3
import os
import aws_cdk as cdk
from hotdog_classifier.training_stack import TrainingStack
from hotdog_classifier.inference_stack import InferenceStack

app = cdk.App()

stage = app.node.try_get_context("stage") or "dev"

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region="us-west-2",
)

TrainingStack(
    app,
    "HotdogTrainingStack" if stage == "prod" else f"HotdogTrainingStack-{stage}",
    stage=stage,
    env=env,
)

InferenceStack(
    app,
    "HotdogInferenceStack" if stage == "prod" else f"HotdogInferenceStack-{stage}",
    stage=stage,
    env=env,
)

app.synth()
