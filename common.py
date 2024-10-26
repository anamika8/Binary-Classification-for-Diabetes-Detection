import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_dataset():
    dataset_file_path = f'diabetes.csv'

    #reading data file into data frame
    df = pd.read_csv(dataset_file_path)
    df.columns = df.columns.str.strip()
    return df