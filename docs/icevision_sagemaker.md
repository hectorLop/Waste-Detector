# Ideas
- Introduction about the project
- Containerizing
- Sagemaker project structure
- Components needed
  - Data on S3
  - Hyperparameters json file
- Dockerfile
- Training files
- Airflow, or may create a new blog post for it?

# Training on SageMaker using Docker and Icevision

## Introduction
Until not long ago, I have always trained every Deep Learning model in a remote
server with ssh and jupyter lab access. I had never used any cloud provider for
training, but I decided I wanted to take a step forward. This decision led me
to use AWS SageMaker in one of my side projects.

Looking into the SageMaker possibilities, I discovered that it allowed to
use custom Docker containers to create training jobs. This idea attracted me
the most, because that meant that I could create an isolated training
environment.

Initially I struggled a lot reading the documentation and several blog posts
until I understood how to create this custom Docker image to be executed
in SageMaker. In this blog bost I will walk you through how to create a 
SageMaker training job that uses a custom Docker image!.

## AWS SageMaker structure
Let's take a look at the structure that will have our training job before
creating any training script nor Docker image.

SageMajer invokes the training code by running a version of the following
command:

```bash
docker run <image> train
```

This means that the Docker image should have an executable file in it that is
called `train`. That will be our training script. Besides, the training
data must be stored into a S3 bucket, so SageMaker can downloads it.

SageMaker uses the following project structure:
```
/opt/ml
├── code
│   ├── train.py
│   └── <other training files>   
│
├── input
│   ├── config
│   │   └── hyperparameters.json
│   │  
│   └── data
│       └── <channel_name>
│           └── <input data>
├── model
│   └── <model files>
└── output
    └── failure
```

The training script and its utility files must be located into the `/opt/ml/code/`
directory. In the other hand, the `input` directory contains both the 
hyperparamenters in a JSON file under the `input/config/` directory and the
training data under the `input/data/<channel name>/` directory. The channel
name could be whatever we want, in out case we will use `training`.
The `model` directory contains any training output or checkpoint.

## Training script
Given that the project is an object detection application, the training script
uses the great Icevision framework.

```
def get_data_loaders(model_type, config) -> Tuple[DataLoader]:
    """
    Get the dataloaders for each set.
    Args:
        annotations (str): Annotations filepath.
        img_dir (str): Images filepath.
        config (Config): Config object.
    Returns:
        Tuple[DataLoader]: Tuple containing:
            - (DataLoader): Training dataloader
            - (DataLoader): Validation dataloader
            - (DataLoader): Test dataloader
    """
    with open('/opt/ml/input/data/training/data/indices.json', 'r') as file:
        indices_dict = json.load(file)

    parser = COCOBBoxParser(annotations_filepath='/opt/ml/input/data/training/data/mixed_annotations.json',
                            img_dir='/opt/ml/input/data/training/')
    splitter = FixedSplitter(splits=[indices_dict['train'], indices_dict['val']])

    train_records, val_records = parser.parse(data_splitter=splitter, autofix=True)
    
    img_size = int(config['img_size'])
    train_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()]) 

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(val_records, valid_tfms)

    # Data Loaders
    train_dl = model_type.train_dl(
        train_ds, batch_size=int(config['batch_size']), num_workers=4, shuffle=True
    )
    valid_dl = model_type.valid_dl(
        valid_ds, batch_size=int(config['batch_size']), num_workers=4, shuffle=False
    )

    return train_dl, valid_dl

def train(config: Dict) -> None:
    """
    Trains a waste detector model.
    Args:
        parameters (Dict): Dictionary containing training parameters.
    """
    model_type = get_object_from_str(config['model_type'])
    backbone = get_object_from_str(config['backbone'])

    train_dl, valid_dl = get_data_loaders(model_type, config)

    extra_args = {
        'img_size': int(config['img_size'])
    }
    
    print("Getting the model")
    model = model_type.model(
        backbone=backbone(pretrained=True),
        num_classes=int(config['num_classes']),
        **extra_args
    )

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='/opt/ml/model/',
        filename=f'sagemaker_model_v{new_version}',
        save_top_k=1,
        verbose=True,
        monitor="valid/loss",
        mode="min",
    )
    
    metrics_callback = MetricsCallback()
    lightning_model = EfficientDetModel(model=model, optimizer=torch.optim.SGD,
                                        learning_rate=float(config['learning_rate']),
                                        metrics=metrics)
    
    print("TRAINING")
    for param in lightning_model.model.parameters():
        param.requires_grad = True
        
    trainer = Trainer(max_epochs=int(config['epochs']), gpus=1,
                      callbacks=[checkpoint_callback, metrics_callback])
    trainer.fit(lightning_model, train_dl, valid_dl)
    
    print('Training complete')

if __name__ == "__main__":
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as json_file:
        hyperparameters = json.load(json_file)

    train(hyperparameters)
    sys.exit(0)
```
