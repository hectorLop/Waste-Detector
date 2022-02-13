# Airflow Operators
import airflow
from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

# Airflow Sagemaker Operators
from airflow.providers.amazon.aws.operators.sagemaker_training import SageMakerTrainingOperator
from airflow.providers.amazon.aws.operators.sagemaker_endpoint import SageMakerEndpointOperator
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook

# AWS SDK for Python
import boto3
import json
import sagemaker as sage

# Amazon SageMaker SDK
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator
from sagemaker.session import s3_input

# Airflow SageMaker Configuration
from sagemaker.workflow.airflow import training_config
from sagemaker.workflow.airflow import model_config_from_estimator
from sagemaker.workflow.airflow import deploy_config_from_estimator


def validate_classifier():
    sess = sage.Session()

    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name

    data_location = 's3://waste-detector/'

    image = f'{account}.dkr.ecr.{region}.amazonaws.com/classifier_validation:latest'


    with open('dags/classifier_hyperparameters.json', 'r') as file:
        hyperparameters = json.load(file)

    model = sage.estimator.Estimator(image,
                role='AWSSagemakerFull-Default',
                instance_count=1,
                instance_type='ml.m5.xlarge',
                hyperparameters=hyperparameters,
                base_job_name='classifier-job',
                sagemaker_session=sess)

    data_channels = {
        'training': data_location
    }

    train_config = training_config(estimator=model,
                                   inputs=data_channels)

    return train_config


def validate_detector():
    sess = sage.Session()

    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name

    data_location = 's3://waste-detector/'

    image = f'{account}.dkr.ecr.{region}.amazonaws.com/waste_validation:detector_latest'


    with open('dags/detector_hyperparameters.json', 'r') as file:
        hyperparameters = json.load(file)

    model = sage.estimator.Estimator(image,
                role='AWSSagemakerFull-Default',
                instance_count=1,
                instance_type='ml.m5.xlarge',
                hyperparameters=hyperparameters,
                base_job_name='detector-job',
                sagemaker_session=sess)

    data_channels = {
        'training': data_location
    }

    train_config = training_config(estimator=model,
                                   inputs=data_channels)

    return train_config

args = {"owner": "airflow",  'depends_on_past': False}
classifier_training = validate_classifier()
detector_training = validate_detector()

with DAG(
    dag_id='validation',
    start_date=days_ago(1),
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
) as dag:
    validate_detector = SageMakerTrainingOperator(
        task_id='detector',
        config = detector_training,
        aws_conn_id = "airflow-sagemaker",
        wait_for_completion = True,
        check_interval = 60, #check status of the job every minute
        max_ingestion_time = None #allow training job to run as long as it needs, change for early stop
    )

    validate_classifier = SageMakerTrainingOperator(
        task_id='classifier',
        config=classifier_training,
        aws_conn_id='airflow-sagemaker',
        wait_for_completion=True,
        check_interval=60,
        max_ingestion_time=None
    )

    validate_detector >> validate_classifier
