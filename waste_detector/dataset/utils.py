import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def split_data(annotations_df):
    #ids, categories = annotations_df['image_id'], annotations_df['category_id']
    df_images = annotations_df.groupby(['image_id'], as_index=False).agg({'category_id': 'count'})
    #df_images = grouped_df.sort_values('category_id', ascending=False).reset_index()
    #print(grouped_df)
    # Use the quantiles of amount of annotations to stratify
    df_images_train, df_images_test = train_test_split(df_images['image_id'],
                                                       #stratify=df_images['category_id'],
                                                       test_size=0.2,
                                                       random_state=42)

    # Use the quantiles of amount of annotations to stratify
    df_images_train, df_images_val = train_test_split(df_images['image_id'],
                                                      #stratify=df_images['category_id'],
                                                      test_size=0.2,
                                                      random_state=2021)

    # df_images solo posee el id, cell types y numero de anotaciones
    df_train = annotations_df[annotations_df['image_id'].isin(df_images_train)]
    df_val = annotations_df[annotations_df['image_id'].isin(df_images_val)]
    df_test = annotations_df[annotations_df['image_id'].isin(df_images_test)]

    return df_train, df_val, df_test

def add_background_imgs(annotations_df, background_df):
    max_image_id = annotations_df['image_id'].max()
    new_max_id = max_image_id + len(background_df)
    
    identifiers = np.arange(max_image_id+1, new_max_id+1).astype(int)
    background_df['image_id'] = identifiers
    
    if any(annotations_df['image_id'].isin(background_df['image_id'])):
        raise ValueError('Image identifiers cannot be duplicated')
        
    annotations_df = pd.concat([annotations_df, background_df], axis=0, ignore_index=True)
    
    return annotations_df