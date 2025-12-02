from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 

# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 

# metadata 
print("Dataset Metadata:")
print(student_performance.metadata) 
print("\n" + "="*50 + "\n")

# variable information 
print("Variable Information:")
print(student_performance.variables)
print("\n" + "="*50 + "\n")

# Combine features and targets into a single dataframe
full_data = pd.concat([X, y], axis=1)

# Save to CSV
csv_filename = "student_performance.csv"
full_data.to_csv(csv_filename, index=False)

print(f"Dataset saved as '{csv_filename}'")
print(f"Dataset shape: {full_data.shape}")
print(f"Columns: {list(full_data.columns)}")
