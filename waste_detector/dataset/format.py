from typing import Dict, Tuple
from waste_detector.dataset.config import TACO_REPLACEMENTS

import pandas as pd

def get_new_categories(categories_df : pd.DataFrame,
                       replacements : Dict) -> pd.DataFrame:
    """
    Get the new categories and its indices

    Args:
        categories_df (pandas.DataFrame): Catagories DataFrame
        replacements (dict): Dictionary containing the name replacements
            for the old categories.
    
    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The annotations DataFrame
            - dict: Dictionary containing the new categories and its
                indices
    """
    # Copy to avoid modifying the original
    df = categories_df.copy()

    # New column with the replacements
    df['new_cat'] = df['name'].replace(replacements)

    new_categories = {}

    # Assign the old categories to the new ones
    for idx, cat in enumerate(pd.unique(df['new_cat']), 1):
        if cat not in new_categories:
            new_categories[cat] = idx

    return df, new_categories

def process_categories(
    categories_df : pd.DataFrame,
    annotations_df : pd.DataFrame,
    replacements : Dict = TACO_REPLACEMENTS
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the categories of a dataset

    Args:
        categories_df (pandas.DataFrame): Catagories DataFrame
        annotations_df (pandas.DataFrame): Annotations DataFrame
        replacements (dict): Dictionary containing the name replacements
            for the old categories.
    
    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The annotations DataFrame
            - pandas.DataFrame: The categories DataFrame
    """
    df = annotations_df.copy()

    # Replaced categories dataframe and new categories
    categories_df, new_categories = get_new_categories(categories_df,
                                                       replacements)

    categories_replacements = {}

    for cat in new_categories:
        # Get the category ids attached at a old category
        old_categories = categories_df[categories_df['new_cat'] == cat]['id']

        # Assigng the old identifiers the identifier to the new category
        for old_cat in old_categories:
            if old_cat not in categories_replacements:
                categories_replacements[old_cat] = new_categories[cat]

    # Make the replacement into the annotation
    df['category_id'] = df['category_id'].replace(categories_replacements)  

    return df, categories_df                                  