algorithm_name=waste_validation
region=eu-west-1

account=$(aws sts get-caller-identity --query Account --output text)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:detector_latest"

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

#$(aws ecr get-login --region ${region} --no-include-email)
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker build -t ${algorithm_name} -f val.Dockerfile .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
