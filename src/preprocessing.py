import pandas as pd
import numpy as np
import datetime

def preprocess_data(df):
    # Drop irrelevant columns
    df.drop(['Unnamed: 0', 'New_Price'], axis=1, inplace=True)

    # Extract manufacturer
    manufacturer = df['Name'].str.split(' ').str.slice(0, 2)
    df['Manufacturer'] = manufacturer.str.join(' ')

    # Calculate age
    curr_year = datetime.datetime.now().year
    df['Year Used'] = df['Year'].apply(lambda x: curr_year - x)

    # Drop original columns
    df.drop(['Name', 'Year'], axis=1, inplace=True)

    # Convert and fill Mileage
    mileage = df['Mileage'].str.split(' ', expand=True)
    df['Mileage'] = pd.to_numeric(mileage[0], errors='coerce')
    df['Mileage'].fillna(df['Mileage'].mean(), inplace=True)

    # Convert and fill Engine
    engine = df['Engine'].str.split(' ', expand=True)
    df['Engine'] = pd.to_numeric(engine[0], errors='coerce')
    df['Engine'].fillna(df['Engine'].mean(), inplace=True)

    # Convert and fill Power
    power = df['Power'].str.split(' ', expand=True)
    df['Power'] = pd.to_numeric(power[0], errors='coerce')
    df['Power'].fillna(df['Power'].mean(), inplace=True)

    # Fill missing seats
    df['Seats'].fillna(df['Seats'].mean(), inplace=True)

    # Remove rare manufacturers
    manufacturer_counts = df['Manufacturer'].value_counts()
    common_manufacturers = manufacturer_counts[manufacturer_counts >= 5].index
    df = df[df['Manufacturer'].isin(common_manufacturers)]

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Manufacturer', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Location'], drop_first=True)

    return df
