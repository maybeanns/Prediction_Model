import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# LabelEncoder is s used to encode categorical variables (like text labels) into numbers.
# This is often necessary because many machine learning algorithms require numerical input data.
# It assigns a unique integer to each unique category in the column. The integers are assigned based on the alphabetical order of the categories.

# Load the data
df = pd.read_csv('islamabad_properties2.csv')

#******************************************* 1. Clean and normalize the data

# Function to convert price to millions
def convert_price_to_millions(price_str):
    num = float(price_str.split()[0])
    if 'Crore' in price_str:
        return num * 10
    elif 'Lakh' in price_str:
        return num * 0.1
    elif 'Arab' in price_str:
        return num * 100
    else:
        return num

# Apply price conversion
df['Price_Millions'] = df['Price'].apply(convert_price_to_millions)

#******************************************** 2. Convert textual data to numerical

# Convert beds to numerical
# You first need to convert the beds columns to string then use this next line or you can just use the original columns if you dont have any string but just to be safe convert it twice
df['Beds'] = df['Beds'].astype(str)
df['Beds_Num'] = df['Beds'].str.extract('(\d+)').astype(float)

# Function to convert area to square feet
def convert_area_to_sqft(area_str):
    num = float(area_str.split()[0])
    unit = area_str.split()[1].lower()
    if 'kanal' in unit:
        return num * 4500  # 1 Kanal = 4500 sq ft
    elif 'marla' in unit:
        return num * 225   # 1 Marla = 225 sq ft
    elif 'acre' in unit:
        return num * 43560 # 1 Acre = 43560 sq ft
    elif 'sqft' in unit or 'sq.ft' in unit:
        return num
    else:
        return np.nan

# Apply area conversion
df['Area_SqFt'] = df['Area'].apply(convert_area_to_sqft)

# 3. Handle missing values

# Check for missing values
print(df.isnull().sum())

# Fill missing values or drop them based on your preference
df = df.dropna()
# Alternatively, you could fill missing values:
# df['Beds_Num'].fillna(df['Beds_Num'].median(), inplace=True)
# df['Area_SqFt'].fillna(df['Area_SqFt'].median(), inplace=True)

# 4. Encode categorical variables

# Encode location
le = LabelEncoder()
df['Location_Encoded'] = le.fit_transform(df['Location'])

# You might want to keep track of the encoding
location_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Location Encoding:")
print(location_mapping)

# 5. Final touches

# Select the features you want to use for your model
features = ['Price_Millions', 'Beds_Num', 'Area_SqFt', 'Location_Encoded']
df_clean = df[features]

# Check the cleaned dataframe
print(df_clean.head())
print(df_clean.describe())

# Save the cleaned data
df_clean.to_csv('clean_islamabad_properties.csv', index=False)

print("Data preparation completed. Cleaned data saved to 'clean_islamabad_properties.csv'")