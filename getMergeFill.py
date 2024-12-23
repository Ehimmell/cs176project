import pandas as pd
import numpy as np

def getData(cali_path, melbourne_path, portugal_path, nrows=1000000):
    """
    cali-path: str, path to california dataset
    melbourne-path: str, path to melbourne dataset
    portugal-path: str, path to portugal dataset
    """

    # Read the data from all three CSV files
    cali = pd.read_csv(cali_path, nrows=nrows)
    melbourne = pd.read_csv(melbourne_path, nrows=nrows)
    portugal = pd.read_csv(portugal_path, nrows=nrows)

    # Map columns for each dataset to make universal names
    cali_column_mapping = {
        'price': 'price',
        'city': 'city',
        'state': 'state',
        'streetAddress': 'street_address',
        'bedrooms': 'bedrooms',
        'bathrooms': 'bathrooms',
        'buildingArea': 'building_area',
        'livingArea': 'living_area',
        'lotAreaUnits': 'lot_size',
        'yearBuilt': 'year_built',
        'longitude': 'longitude',
        'latitude': 'latitude',
        'parking': 'parking_spaces',
        'garageSpaces': 'garage_spaces',
        'levels': 'levels',
        'pool': 'pool',
        'homeType': 'home_type',
        'country': 'country',
    }

    melbourne_column_mapping = {
        'Price': 'price',
        'Suburb': 'city',
        'Address': 'street_address',
        'Bedroom2': 'bedrooms',
        'Bathroom': 'bathrooms',
        'BuildingArea': 'building_area',
        'Landsize': 'lot_size',
        'YearBuilt': 'year_built',
        'Longtitude': 'longitude',
        'Lattitude': 'latitude',
        'Car': 'parking_spaces',
        'Type': 'home_type',
    }

    portugal_column_mapping = {
        'Price': 'price',
        'City': 'city',
        'Town': 'town',
        'Type': 'home_type',
        'NumberOfBedrooms': 'bedrooms',
        'NumberOfBathrooms': 'bathrooms',
        'BuiltArea': 'building_area',
        'LivingArea': 'living_area',
        'LotSize': 'lot_size',
        'ConstructionYear': 'year_built',
        'Parking': 'parking_spaces',
        'Garage': 'garage_spaces',
        'Floor': 'levels',
        'TotalRooms': 'total_rooms',
    }

    # Rename columns to make them universal
    cali.rename(columns=cali_column_mapping, inplace=True)
    melbourne.rename(columns=melbourne_column_mapping, inplace=True)
    portugal.rename(columns=portugal_column_mapping, inplace=True)

    # Add a column to each dataset to identify which dataset it came from
    cali['frame'] = 'california'
    melbourne['frame'] = 'melbourne'
    portugal['frame'] = 'portugal'

    # Create a set of all columns from all three datasets
    all_columns = set(cali.columns) | set(melbourne.columns) | set(portugal.columns)

    # Ensure all dataframes have the same columns
    for df in [cali, melbourne, portugal]:
        for col in all_columns:
            if col not in df.columns:
                df[col] = np.nan

    # Concatenate the three datasets
    merged_data = pd.concat([cali, melbourne, portugal], ignore_index=True, sort=False)

    # Define a list of columns to drop
    columns_to_drop = [
        'id', 'ElectricCarsCharging', 'PublishDate', 'EnergyEfficiencyLevel', 'SellerG',
        'ConservationStatus', 'latitude', 'longitude', 'GrossArea', 'HasParking',
        'Propertycount', 'Regionname', 'CouncilArea', 'Date', 'country', 'Unnamed: 0',
        'datePostedString', 'is_bankOwned', 'is_forAuction', 'event', 'time', 'state', 'zipcode',
        'hasBadGeocode', 'description', 'currency', 'livingAreaValue', 'hasGarage', 'pool',
        'spa', 'isNewConstruction', 'hasPetsAllowed', 'county', 'Rooms', 'Method', 'Distance',
        'Postcode', 'NumberOfWC', 'lot_size', 'building_area'
    ]

    # Drop the columns
    existing_columns_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
    merged_data.drop(columns=existing_columns_to_drop, inplace=True)

    # Fill in missing values with median for continuous columns and mode for discrete
    merged_data['parking_spaces'] = merged_data['parking_spaces'].fillna(0)
    merged_data['year_built'] = merged_data['year_built'].replace(0, np.nan)
    merged_data['year_built'] = merged_data['year_built'].fillna(merged_data['year_built'].median())
    merged_data['price'] = merged_data['price'].fillna(merged_data['price'].median())
    merged_data['TotalArea'] = merged_data['TotalArea'].fillna(merged_data['TotalArea'].median())
    merged_data['Elevator'] = merged_data['Elevator'].fillna(False)
    merged_data['District'] = merged_data['District'].fillna('Unknown')
    merged_data['bathrooms'] = merged_data['bathrooms'].fillna(merged_data['bathrooms'].median())
    merged_data['bedrooms'] = merged_data['bedrooms'].fillna(merged_data['bedrooms'].median())
    merged_data['total_rooms'] = merged_data['total_rooms'].fillna(
        merged_data['bathrooms'] + merged_data['bedrooms']
    )
    merged_data['street_address'] = merged_data['street_address'].fillna('Unknown')
    merged_data['garage_spaces'] = merged_data['garage_spaces'].fillna(0)
    merged_data['living_area'] = merged_data['living_area'].fillna(merged_data['living_area'].median())
    merged_data['levels'] = merged_data['levels'].fillna('One Story')
    merged_data['EnergyCertificate'] = merged_data['EnergyCertificate'].fillna(
        merged_data['EnergyCertificate'].mode()[0]
    )

    # Return the merged data
    return merged_data
