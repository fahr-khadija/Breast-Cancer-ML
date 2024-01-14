import pandas as pd

# Load the CSV file
file_path = "breast_cancer_data.csv"
df = pd.read_csv(file_path)

# Split the DataFrame into two parts
df_part1 = df.head(350)
df_part2 = df.iloc[350:]

# Save the two parts into separate CSV files
part1_file_path = "breast_cancer_data_load.csv"
part2_file_path = "breast_cancer_data_train.csv"

df_part1.to_csv(part1_file_path, index=False)
df_part2.to_csv(part2_file_path, index=False)

print(f"First 350 rows saved to {part1_file_path}")
print(f"Remaining rows saved to {part2_file_path}")
