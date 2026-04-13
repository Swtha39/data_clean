import pandas as pd
import numpy as np

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("used_cars.csv")   # change file name if needed

# -------------------------------
# Task 1 — Explore Data
# -------------------------------
print("Shape of dataset:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())

# Identify issues manually (print observations)
print("\n--- Data Quality Issues Observed ---")
print("1. Missing values present in some columns")
print("2. 'mileage' column has non-numeric values (e.g., '18 kmpl')")
print("3. 'brand' column has inconsistent formatting (uppercase/lowercase/spaces)")
print("4. Possible duplicate rows")

# -------------------------------
# Task 2 — Data Cleaning
# -------------------------------

# 1. Drop rows where target (selling_price) is null
df = df.dropna(subset=['selling_price'])

# 2. Fill missing numerical values with mean
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 3. Fill missing categorical values with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Clean 'brand' column (strip + lowercase)
if 'brand' in df.columns:
    df['brand'] = df['brand'].str.strip().str.lower()

# 5. Extract numeric values from 'mileage'
if 'mileage' in df.columns:
    df['mileage'] = df['mileage'].astype(str).str.extract('(\d+\.?\d*)')
    df['mileage'] = df['mileage'].astype(float)

# 6. Remove duplicates
df = df.drop_duplicates()

print("\nData cleaned successfully!")

# -------------------------------
# Task 3 — Baseline MAE
# -------------------------------

# Calculate mean selling price
mean_price = df['selling_price'].mean()

# Predict mean for all rows
df['baseline_pred'] = mean_price

# Calculate MAE
mae = np.mean(np.abs(df['selling_price'] - df['baseline_pred']))

print("\n--- Baseline Model ---")
print("Mean Selling Price:", mean_price)
print("Baseline MAE:", mae)