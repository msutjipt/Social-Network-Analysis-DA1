import pandas as pd

def inspect_data():
    filepath_dataset = 'YOUR_PATH'
    filepath_graph = 'YOUR_PATH'

    df_dataset = pd.read_csv(filepath_dataset, delimiter=";")
    df_graph = pd.read_csv(filepath_graph, delimiter=",")

    print(df_graph)
    print(df_dataset)


if __name__=='__main__':
    inspect_data()

