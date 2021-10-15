import json


def load_cfg(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
