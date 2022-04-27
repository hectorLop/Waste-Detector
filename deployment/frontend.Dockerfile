FROM ubuntu:20.04

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
    pip install boto3 && \
    pip install numpy && \
    pip install scipy

ARG AWS_ACCESS_KEY_ID_ARG
ARG AWS_SECRET_ACCESS_KEY_ARG
ARG AWS_DEFAULT_REGION_ARG

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_ARG
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_ARG
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_ARG

# copy the scripts inside the container
COPY frontend.py /deployment/frontend.py
COPY utils.py /deployment/utils.py

EXPOSE 8501

COPY example_imgs/ /deployment/example_imgs/

ENTRYPOINT ["python3","-m", "deployment.frontend"]
