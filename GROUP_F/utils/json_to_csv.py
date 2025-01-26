import json
from pandas import json_normalize
import pandas as pd


def json_to_csv():

    filepath = 'YOUR_PATH'
    with open(filepath) as f:
        data = json.load(f)

    df = json_normalize(data)

    df.to_csv("dataset.csv", index=False, sep=",")


if __name__ == '__main__':
    json_to_csv()