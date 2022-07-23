FROM python:3.9

# Install dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install libgtk2.0-dev -y

RUN pip install gradio && \
    pip install matplotlib && \
    pip install Pillow && \
    pip install Jinja2 && \
    pip install opencv-python && \
    pip install numpy && \
    pip install scipy

# copy the scripts inside the container
COPY frontend.py /deployment/frontend.py
COPY utils.py /deployment/utils.py

EXPOSE 8501

COPY example_imgs/ /deployment/example_imgs/

ENTRYPOINT ["python3","-m", "deployment.frontend"]
