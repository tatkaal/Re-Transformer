import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame
df = pd.read_csv('merged_parallel.csv')

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

df = df[:6000]

# Split the DataFrame
df_valid_test, df_train = train_test_split(df, test_size=(1 - 2 * 0.01), random_state=42)
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42)

# Save the DataFrames to new CSV files
df_train.to_csv('training.csv', index=False)
df_valid.to_csv('validation.csv', index=False)
df_test.to_csv('testing.csv', index=False)