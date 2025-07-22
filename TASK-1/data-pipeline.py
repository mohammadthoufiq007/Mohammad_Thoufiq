import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def fetch_data(file_path):
    """Fetches data from a CSV file"""
    print("Fetching data...")
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocesses the data by applying scaling, encoding, and imputation"""
    print("Preprocessing data...")

    X = df.copy()
    y = None  

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    X_transformed = preprocessor.fit_transform(X)


    X_transformed_df = pd.DataFrame(X_transformed)

    return X_transformed_df, y


def save_data(X, y=None, output_path="processed-data.csv"):
    """Saves the preprocessed data to a CSV file"""
    print("Saving data...")
    df = X if y is None else pd.concat([X, y.reset_index(drop=True)], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def execute_pipeline(input_csv):
    """Runs the entire data pipeline"""
    df = fetch_data(input_csv)
    X_transformed, y = preprocess_data(df)
    save_data(X_transformed, y)


if __name__ == "__main__":
    INPUT_CSV = "data.csv"  
    if not os.path.exists(INPUT_CSV):
        print(f"File '{INPUT_CSV}' not found. Please add your dataset.")
    else:
        execute_pipeline(INPUT_CSV)
