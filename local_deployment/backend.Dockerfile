FROM python:3.9

RUN apt-get update && apt-get -y install libgl1

RUN apt-get install -y git

RUN git config --global url."https://".insteadOf git://

RUN pip install -e git://github.com/hectorLop/icevision.git@aws-lambda-5#egg=icevision[all] --upgrade -q

RUN pip install pandas && \
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
