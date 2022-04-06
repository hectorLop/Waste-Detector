#FROM ubuntu:20.04
FROM public.ecr.aws/lambda/python:3.8

RUN yum -y update
RUN yum install -y mesa-libGL
RUN yum -y install gcc
RUN yum install -y git

RUN git config --global url."https://".insteadOf git://

RUN pip install -e git://github.com/hectorLop/icevision.git@aws-lambda-5#egg=icevision[all] --upgrade -q

RUN pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
    pip install mmcv==1.3.17 && \
    pip install Pillow 

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY utils.py ./
COPY model.py ./
COPY classifier.py ./
##COPY example_imgs/* /deployment/example_imgs/
#
COPY app.py ./ 
#COPY efficientDet_icevision_v9.ckpt ./model_dir/
#COPY class_efficientB0_taco_7_class_v1.pth ./model_dir/
COPY ckpts_download.py ./
##RUN mkdir /deployment/checkpoints
RUN bash -c "python ckpts_download.py"
#CMD [ "python", "ckpts_download.py" ]
#CMD ["app.get_models"]
CMD ["app.handler"]

#ENTRYPOINT ["python3", "-m", "uvicorn", "deployment.backend:app", "--host", "0.0.0.0", "--port", "5000"]
