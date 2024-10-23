import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset():
    dataset_file_path = f'diabetes.csv'

    #reading data file into data frame
    df = pd.read_csv(dataset_file_path)
    df.columns = df.columns.str.strip()
    return df

def feature_scaling(df):
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    scaler = StandardScaler()

    # Fit and transform the scaler on the features
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def preprocess_data(df):
    columns_with_comma_decimal = ['chol_hdl_ratio', 'bmi', 'waist_hip_ratio']
    for column in columns_with_comma_decimal:
        df[column] = df[column].astype(str).str.replace(',', '.').astype(float)

    # Encode the 'gender' column
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})

    # Encode the 'diabetes' column
    df['diabetes'] = df['diabetes'].map({'No diabetes': 0, 'Diabetes': 1})

    # Handle Missing Values (if any)
    df.fillna(df.median(), inplace=True)
    scaled_data, y = feature_scaling(df)
    return scaled_data, y

def get_train_test_data():
    scaled_data, y = preprocess_data(df=load_dataset())
    '''
    using stratify to split the data in such a way that the proportion of diabetes vs no diabetes cases in
    both training and testing sets will match the proportion in the full dataset y.
    '''
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, y


if __name__ == "__main__":
    load_dataset()