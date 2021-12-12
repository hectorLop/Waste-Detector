import json
import argparse
import yaml
import pandas as pd
import numpy as np

def process_annotations(parameters):
    annotations_df = pd.DataFrame()
    images_df = pd.DataFrame()

    categories = [
        {'id': 0, 'name': 'Background', 'supercategory': 'Background'}, 
        {'id': 1, 'name': 'Waste', 'supercategory': 'Waste'}
    ]
    
    for idx, element in enumerate(parameters['annotations']):
        file, img_path, contains_background = element

        with open(file, 'r') as file:
            data = json.load(file)

        temp_annotations_df = pd.DataFrame(data['annotations'])
        temp_images_df = pd.DataFrame(data['images'])[['id', 'width', 'height', 'file_name']]

        if idx == 0:
            max_id = temp_images_df['id'].max()
        else:
            max_id = images_df['id'].max()

        if idx != 0:
            new_ids = np.arange(max_id + 1, max_id + len(temp_images_df) + 1)
            old_ids = temp_images_df['id'].sort_values(ascending=True)

            if len(images_df) > 0 and any(images_df['id'].isin(new_ids)):
                raise ValueError('There are image identifiers duplicated')
        
            # Map the old identifiers to the new ones
            map_ids = {old: new for old, new in zip(old_ids, new_ids)}

            # Replace the identifiers
            temp_images_df['id'] = temp_images_df['id'].replace(map_ids)
            temp_annotations_df['image_id'] = temp_annotations_df['image_id'].replace(map_ids)

        # Set in file_name the full image path
        temp_images_df['file_name'] = img_path + '/' + temp_images_df['file_name']

        if contains_background:
            temp_annotations_df[temp_annotations_df['category_id'] != 0]['category_id'] = 1
        else:
            temp_annotations_df['category_id'] = 1

        annotations_df = pd.concat([annotations_df, temp_annotations_df], ignore_index=True)
        images_df = pd.concat([images_df, temp_images_df], ignore_index=True)

    return annotations_df, images_df, categories

def create_new_annotation_file(new_file, annotations, images, categories):
    annotations_dict = {}
    annotations_dict['annotations'] = annotations.to_dict(orient='records')
    annotations_dict['images'] = images.to_dict(orient='records')
    annotations_dict['categories'] = categories

    with open(new_file, 'w') as new_file:
        json.dump(annotations_dict, new_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    annotations, images, categories = process_annotations(params)
    create_new_annotation_file(params['filename'], annotations, images, categories)