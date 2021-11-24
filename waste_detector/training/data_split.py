import pandas as pd

from sklearn.model_selection import train_test_split

def split_data(annotations_df):
    grouped_df = annotations_df.groupby(['image_id']).agg({'category_id': 'count'})
    df_images = grouped_df.sort_values('category_id', ascending=False).reset_index()

    # Use the quantiles of amount of annotations to stratify
    df_images_train, df_images_test = train_test_split(df_images,
                                                    #stratify=df_images['bbox'],
                                                    test_size=0.2,
                                                    random_state=42)

    # Use the quantiles of amount of annotations to stratify
    df_images_train, df_images_val = train_test_split(df_images_train,
                                                    #stratify=df_images_train['bbox'],
                                                    test_size=0.2,
                                                    random_state=2021)

    # df_images solo posee el id, cell types y numero de anotaciones
    df_train = annotations_df[annotations_df['image_id'].isin(df_images_train['image_id'])]
    df_val = annotations_df[annotations_df['image_id'].isin(df_images_val['image_id'])]
    df_test = annotations_df[annotations_df['image_id'].isin(df_images_test['image_id'])]

    return df_train, df_val, df_test