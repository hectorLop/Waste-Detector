#FROM ubuntu:20.04
FROM public.ecr.aws/lambda/python:3.8

#RUN apt-get install -y libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev
# Install dependencies
#RUN apt-get update && \
#    apt-get install -y python3-pip && \
#    apt-get install -y libgl1-mesa-glx

#RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
#RUN apt-get install libgtk2.0-dev -y
RUN yum -y update
RUN yum -y install gcc

RUN pip install icevision[all] && \
    pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
#    pip install mmcv-full==1.3.17 && \
    pip install Pillow 
    #pip install fastapi && \
    #pip install "uvicorn[standard]"

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
#COPY backend.py /deployment/backend.py
#COPY utils.py /deployment/utils.py
#COPY model.py /deployment/model.py
#COPY classifier.py /deployment/classifier.py
#COPY example_imgs/* /deployment/example_imgs/

#EXPOSE 5000

COPY app.py ./ 

#RUN mkdir /deployment/checkpoints

#CMD ["app.get_models"]
CMD ["app.handler"]

#ENTRYPOINT ["python3", "-m", "uvicorn", "deployment.backend:app", "--host", "0.0.0.0", "--port", "5000"]
