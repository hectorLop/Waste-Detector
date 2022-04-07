FROM ubuntu:20.04

#RUN apt-get install -y libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y libgl1-mesa-glx

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install libgtk2.0-dev -y

RUN pip install gradio && \
    pip install matplotlib && \
    pip install Pillow && \
    pip install Jinja2 && \
    pip install opencv-python && \
    pip install boto3

# TODO: WANDB key

# copy the training script inside the container
COPY frontend.py /deployment/frontend.py
COPY utils.py /deployment/utils.py

EXPOSE 8501

COPY example_imgs/ /deployment/example_imgs/

ENTRYPOINT ["python3","-m", "deployment.frontend"]
