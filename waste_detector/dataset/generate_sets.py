import json
from typing import Dict
import pandas as pd
import argparse
import yaml

from waste_detector.dataset.format import process_categories
from waste_detector.dataset.data_split import split_data

def aggregate_datasets(annotations_df, images_df):
    data = []

    for id in annotations_df['image_id']:
        image_data = images_df[images_df['id'] == id]
        
        file_name = image_data['file_name'].values[0]
        width= image_data['width'].values[0]
        height = image_data['height'].values[0]

        data.append((file_name, width, height))

    df = pd.DataFrame(data, columns=['filename', 'width', 'height'])
    annotations_df = pd.concat([annotations_df, df], axis=1)

    return annotations_df

def generate_sets(config : Dict):
    with open(config['annotations'], 'r') as file:
        annotations = json.load(file)

    categories_df = pd.DataFrame(annotations['categories'])
    images_df = pd.DataFrame(annotations['images'])
    annotations_df = pd.DataFrame(annotations['annotations'])

    print('Aggregating the datasets')
    annotations_df = aggregate_datasets(annotations_df, images_df)
    print('Processing the new categories')
    annotations_df, categories_df = process_categories(categories_df,
                                                       annotations_df)

    print('Preparing the data')
    train_df, val_df, test_df = split_data(annotations_df)

    train_df.to_csv(config['train_path'], index=False)
    val_df.to_csv(config['val_path'], index=False)
    test_df.to_csv(config['test_path'], index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    generate_sets(params['parameters'])


