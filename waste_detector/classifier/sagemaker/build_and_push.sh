algorithm_name=classifier_training
region=eu-west-1

account=$(aws sts get-caller-identity --query Account --output text)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

$(aws ecr get-login --region ${region} --no-include-email)

docker build -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

docker rmi --force $(docker images -f "dangling=true" -q)