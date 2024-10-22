import pandas as pd

def load_dataset():
    dataset_file_path = f'diabetes.csv'

    #reading data file into data frame
    df = pd.read_csv(dataset_file_path)
    df.columns = df.columns.str.strip()
    return df.head()

if __name__ == "__main__":
    load_dataset()