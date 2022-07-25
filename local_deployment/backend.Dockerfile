FROM python:3.9

RUN apt-get update && apt-get -y install libgl1

RUN pip install icevision[all] && \ 
    pip install pandas && \
    pip install effdet && \
    pip install mmcv==1.3.17 && \
    pip install Pillow && \
    pip install "fastapi[all]" && \
    pip install huggingface_hub

# copy the training script inside the container
COPY utils.py ./
COPY model.py ./
COPY classifier.py ./

COPY app.py ./ 
