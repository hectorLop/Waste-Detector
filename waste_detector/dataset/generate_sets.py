import json
from typing import Dict
import pandas as pd
import argparse
import yaml
import pickle

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

def save_to_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)
        
def aggregate_annotations_files(files):
    categories_df = pd.DataFrame()
    images_df = pd.DataFrame()
    annotations_df = pd.DataFrame()
    
    for file in files:
        with open(file, 'r') as file:
            annotations = json.load(file)
        
        print('Concatenating datasets')
        categories_df = pd.concat([categories_df, pd.DataFrame(annotations['categories'])])
        categories_df = categories_df.reset_index(drop=True)
        
        images_df = pd.concat([images_df, pd.DataFrame(annotations['images'])])
        images_df = images_df.reset_index(drop=True)
        
        annotations_df = pd.concat([annotations_df, pd.DataFrame(annotations['annotations'])])
        annotations_df = annotations_df.reset_index(drop=True)

    print('Aggregating the datasets')
    annotations_df = aggregate_datasets(annotations_df, images_df)
        
    return annotations_df, categories_df

def process_unique_annotations(file):
    with open(file, 'r') as file:
        annotations = json.load(file)

    categories_df = pd.DataFrame(annotations['categories'])
    images_df = pd.DataFrame(annotations['images'])
    annotations_df = pd.DataFrame(annotations['annotations'])

    print('Aggregating the datasets')
    annotations_df = aggregate_datasets(annotations_df, images_df)
    
    return annotations_df, categories_df

def generate_sets(config : Dict):
    if len(config['annotations']) == 1:
        annotations_df, categories_df = process_unique_annotations(config['annotations'])
    else:
        annotations_df, categories_df = aggregate_annotations_files(config['annotations'])
        
    print('Processing the new categories')
    annotations_df, categories_df = process_categories(categories_df,
                                                       annotations_df)

    print('Preparing the data')
    train_df, val_df, test_df = split_data(annotations_df)
    
    with open(config['train_path'], 'wb') as file:
        pickle.dump(train_df, file)
        
    with open(config['val_path'], 'wb') as file:
        pickle.dump(val_df, file)
        
    with open(config['test_path'], 'wb') as file:
        pickle.dump(test_df, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    generate_sets(params)


