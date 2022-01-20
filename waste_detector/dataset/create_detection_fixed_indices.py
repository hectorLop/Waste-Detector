import argparse
import json
from typing import Dict

import pandas as pd
import yaml

from waste_detector.dataset.utils import get_detection_indices

def create_indices(config: Dict):
    with open(config["annotations"], "r") as file:
        annotations = json.load(file)

    annotations_df = pd.DataFrame(annotations["annotations"])

    train, val, test = get_detection_indices(annotations_df)

    data = {
        'train': train,
        'val': val,
        'test': test
    }

    with open(config['output_file'], 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config YAML file")
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    create_indices(params)