name=$1
region=eu-west-1
	
ecs-cli configure --cluster ${name} --default-launch-type FARGATE --config-name ${name} --region ${region}

ecs-cli up --cluster-config ${name}
