algorithm_name=$1
dockerfile=$2
region=eu-west-1

account=$(aws sts get-caller-identity --query Account --output text)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

#$(aws ecr get-login --region ${region} --no-include-email)

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker build -t ${algorithm_name} -f ${dockerfile} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
