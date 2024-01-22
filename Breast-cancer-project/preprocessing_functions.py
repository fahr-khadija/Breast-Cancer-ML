import pandas as pd

def preprocess_data(data):
    expected_column_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness', 'mean concavity',
                              'mean concave points', 'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error', 'perimeter error', 'area error',
                              'smoothness error', 'compactness error', 'concavity error',
                              'concave points error', 'symmetry error', 'fractal dimension error',
                              'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                              'worst smoothness', 'worst compactness', 'worst concavity',
                              'worst concave points', 'worst symmetry', 'worst fractal dimension',
                              'target']

    # Check for null values
    null_values = data.isnull().sum()
    print("Null Values:\n", null_values)

    # Check column names
    column_names = data.columns
    print("\nColumn Names:\n", column_names)

    # Check column types
    column_types = data.dtypes
    print("\nColumn Types:\n", column_types)

    # Check the number of unique values in each column
    for col in data.columns:
        unique_values = data[col].nunique()
        print(f"\nNumber of Unique Values in '{col}': {unique_values}")

    actual_column_names = data.columns.tolist()

    # Check if the actual column names match the expected ones
    if set(actual_column_names) == set(expected_column_names):
        print("Column names match the expected ones.")
    else:
        print("Column names do not match the expected ones.")

    # Return the preprocessed DataFrame
    return data

