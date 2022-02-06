from sagemaker import get_execution_role
import sagemaker as sage
import json
import argparse

def train(name : str):
    # role = get_execution_role()
    sess = sage.Session()

    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name

    data_location = 's3://waste-detector/'

    image = f'{account}.dkr.ecr.{region}.amazonaws.com/classifier_training:latest'

    with open('hyperparameters.json', 'r') as json_file:
        hyperparameters = json.load(json_file)

    model = sage.estimator.Estimator(image,
                role='AWSSagemakerFull-Default',
                # role=role,
                instance_count=1, 
                instance_type='ml.g4dn.xlarge',
                hyperparameters=hyperparameters,
                #  output_path="s3://{}/output".format("slytherins-test"),
                sagemaker_session=sess)

    data_channels = {
        'training': data_location 
    }

    model.fit(data_channels, job_name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Job name")
    args = parser.parse_args()

    train(args.name)