import pandas as pd
import numpy as np
import datetime

def preprocess_data(df):
    df.drop(['Unnamed: 0', 'New_Price'], axis=1, inplace=True)

    # Extract Manufacturer
    manufacturer = df['Name'].str.split(' ').str.slice(0, 2)
    df['Manufacturer'] = manufacturer.str.join(' ')

    # Calculate car age
    current_year = datetime.datetime.now().year
    df['Year Used'] = df['Year'].apply(lambda x: current_year - x)

    df.drop(['Name', 'Year'], axis=1, inplace=True)

    # Clean numeric columns
    df['Mileage'] = pd.to_numeric(df['Mileage'].str.split(' ').str[0], errors='coerce')
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

    df['Engine'] = pd.to_numeric(df['Engine'].str.split(' ').str[0], errors='coerce')
    df['Engine'] = df['Engine'].fillna(df['Engine'].mean())
    
    df['Power'] = pd.to_numeric(df['Power'].str.split(' ').str[0], errors='coerce')
    df['Power'] = df['Power'].fillna(df['Power'].mean())

    df['Seats'] = df['Seats'].fillna(df['Seats'].mean())

    # Filter rare manufacturers
    manufacturer_counts = df['Manufacturer'].value_counts()
    common_manufacturers = manufacturer_counts[manufacturer_counts >= 5].index
    df = df[df['Manufacturer'].isin(common_manufacturers)]

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Manufacturer', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Location'], drop_first=True)
    df.to_csv("data/processed/processed_data.csv", index=False)
    return df
