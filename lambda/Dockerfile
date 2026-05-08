FROM public.ecr.aws/lambda/python:3.12

COPY predict.py ${LAMBDA_TASK_ROOT}/

CMD ["predict.handler"]
