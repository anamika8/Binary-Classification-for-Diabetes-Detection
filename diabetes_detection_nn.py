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

def feature_scaling(df):
    # Check if 'diabetes' column exists, drop it if present
    if 'diabetes' in df.columns:
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
    else:
        X = df
        y = None

    scaler = StandardScaler()

    # Fit and transform the scaler on the features
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def preprocess_data(df):
    df = df.drop('patient_number', axis=1)
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
    print("Sample training data: \n", X_train[:5])
    print("Sample testing data: \n", X_test[:5])
    return X_train, X_test, y_train, y_test, y

def train_NN_model():
    # getting the train data
    X_train, X_test, y_train, y_test, y = get_train_test_data()
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(X_train.shape[1],)),
        Dense(64, activation='tanh'),
        Dense(1, activation='sigmoid')  # Since it's a binary classification
    ])
    # compile the model
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    # Training the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    return model


if __name__ == "__main__":
    train_NN_model()