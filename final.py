import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
DATA MERGING AND CLEANING

Merge the data a few different ways, and fill na's
'''
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

'''
ADDITIONAL DATA MERGING FOR EXPLORATION

Try merging two dataframes instead of all three and do an inner merge on price after rounding it to remove outlier prices
'''

def additionalMerge(cali_path, portugal_path, melbourne_path, nrows = 1000000):
    cali = pd.read_csv(cali_path, nrows=nrows)
    melbourne = pd.read_csv(melbourne_path, nrows=nrows)
    portugal = pd.read_csv(portugal_path, nrows=nrows)

    caliPortugal = pd.merge(cali, portugal, how='outer')
    caliMelbourne = pd.merge(cali, melbourne, how='outer')
    portugalMelbourne = pd.merge(portugal, melbourne, how='outer')

    cali['price'] = cali['price'].map(lambda x: x / 1000)
    melbourne['price'] = melbourne['price'].map(lambda x: x / 1000)
    portugal['price'] = portugal['price'].map(lambda x: x / 1000)

    priceExemplars = pd.merge(cali, melbourne, how='outer', on='price')
    priceExemplars = pd.merge(priceExemplars, portugal, how='inner', on='price')

    return caliPortugal, caliMelbourne, portugalMelbourne, priceExemplars

'''
PIVOTING, STACKING, AND MORE MERGING

Pivot and stack to explore the relationship between cities and prices
'''

def pivotAndStack(cali_path, melbourne_path, portugal_path, nrows=1000000):

    data = getData(cali_path, melbourne_path, portugal_path, nrows=nrows)

    data = cleansortMergedData(data)

    # 1. Multi-Level Pivot Table with Multiple Aggregations
    pivot_table = data.pivot_table(
        index='city',  # Rows: City
        columns='frame',  # Columns: Dataset source (California, Melbourne, Portugal)
        values=['price', 'bedrooms', 'bathrooms'],  # Aggregated columns
        aggfunc={
            'price': 'mean',  # Average price
            'bedrooms': 'median',  # Median number of bedrooms
            'bathrooms': 'sum'  # Total number of bathrooms
        }
    )

    # 2. Stacking the Multi-Level Pivot Table
    stacked_table = pivot_table.stack()

    # 3. Custom Calculations: Price per Bedroom
    data['price_per_bedroom'] = data['price'] / data['bedrooms']
    pivot_price_per_bedroom = data.pivot_table(
        index='city',
        columns='frame',
        values='price_per_bedroom',
        aggfunc='mean'
    )

    # 4. Merge Stacked Data with Additional Metrics
    # Calculate average price by frame (dataset source)
    avg_price_by_frame = data.groupby('frame')['price'].mean()
    stacked_table_reset = stacked_table.reset_index()  # Reset index for merging
    merged_data = stacked_table_reset.merge(
        avg_price_by_frame, how='left', left_on='frame', right_index=True, suffixes=('', '_avg_frame')
    )

    return pivot_table, stacked_table, pivot_price_per_bedroom, merged_data

'''
CLEANING AND SORTING

Clean the data further and sort it 
'''


def cleansortMergedData(merged):
    # Get merged data from getData. paths are variable, need to adjust accordingly

    # Drop unnecessary columns
    columns_to_drop = ['stateId', 'countyId', 'cityId', 'pricePerSquareFoot', 'EnergyCertificate'
        , 'street_address', 'District', 'town', 'city', 'District', 'total_rooms']
    merged.drop(columns_to_drop, axis=1, inplace=True)

    ## CLEANING COLUMN "HOME_TYPE"
    # Group all values into Land', 'House', 'Multi-Unit Housing', omitting those that don't fit within those 3 groups
    # 'Multi-Unit Housing' is the all-encompassing term for all Apartments/Duplexes/Units due to ambiguity in Melbourne data

    ## PORTUGAL HOME_TYPE CLEANUP
    # Dropping rows where 'home_type' is NOT 'House' or 'Duplex' or 'Apartment' or 'Land' from Portugal data
    portugal_hometype_drop = ['Farm', 'Store', 'Other - Residential', 'Building',
                              'Transfer of lease', 'Garage', 'Other - Commercial',
                              'Warehouse', 'Investment', 'Hotel', 'Office',
                              'Storage', 'Industrial', 'Studio', 'Estate', 'Manor']
    merged.drop(merged[merged['home_type'].isin(portugal_hometype_drop)].index, inplace=True)

    # Mansions and Manors get merged with Houses, and 'Apartment' and 'Duplex' with 'Multi-Unit Housing'
    merged['home_type'] = merged['home_type'].replace('Mansion', 'House')
    merged['home_type'] = merged['home_type'].replace('Manor', 'House')
    merged['home_type'] = merged['home_type'].replace('Apartment', 'Multi-Unit Housing')
    merged['home_type'] = merged['home_type'].replace('Duplex', 'Multi-Unit Housing')

    ## CALIFORNIA HOME_TYPE CLEANUP
    # Houses, including Townhouses, get grouped under 'House',
    # while multi-family houses, condos, apartments get grouped under 'Multi-Unit Housing'.
    # 'LOT' gets renamed to 'Land' for unity.

    cali_repl = {
        'SINGLE_FAMILY': 'House',
        'TOWNHOUSE': 'House',
        'MULTI_FAMILY': 'Multi-Unit Housing',
        'LOT': 'Land',
        'CONDO': 'Multi-Unit Housing',
        'APARTMENT': 'Multi-Unit Housing'
    }
    merged['home_type'] = merged['home_type'].replace(cali_repl)

    ## MELBOURNE HOME_TYPE CLEANUP
    # 'h' gets renamed with 'House', 't' which stands for townhouse also gets renamed 'House',
    # while 'u' standing for unit go under 'Multi-Unit Housing'

    merged['home_type'] = merged['home_type'].replace('h', 'House')
    merged['home_type'] = merged['home_type'].replace('t', 'House')
    merged['home_type'] = merged['home_type'].replace('u', 'Multi-Unit Housing')

    ## DROP NA
    merged.dropna(subset=['home_type'], inplace=True)

    ## PARKING CLEANUP

    # Drop Houses/Multi-Unit Housing with no bed/bath (Not housing - likely parking lots/towers)
    merged.drop(merged[((merged['home_type'] == 'House') |
                        (merged['home_type'] == 'Multi-Unit Housing')) &
                       (merged['bedrooms'] == 0) &
                       (merged['bathrooms'] == 0)
                       ].index, inplace=True)

    # Replace False values in garage_spaces to 0, and True values default to 1
    merged['garage_spaces'] = merged['garage_spaces'].replace(False, 0.0)
    merged['garage_spaces'] = merged['garage_spaces'].replace(True, 1.0)

    ## Merging parking_spaces and garage_spaces for clarity
    # Convert parking_spaces and garage_spaces to numeric values, preparing for merge
    merged['parking_spaces'] = pd.to_numeric(merged['parking_spaces'])
    merged['garage_spaces'] = pd.to_numeric(merged['garage_spaces'])

    # Add the two columns together, add back into parking_spaces
    merged['parking_spaces'] = merged['parking_spaces'] + merged['garage_spaces']

    # Drop garage_spaces column
    merged.drop('garage_spaces', axis=1, inplace=True)

    ## LEVELS CLEANUP: 3 parts - by Land, House, Multi-Unit Housing.
    ## Land is 0 for levels, levels for House means how many stories,
    ## levels for Multi-Unit Housing signifies level the unit is located on- with 1 being on the ground

    # LAND CLEANUP
    # Drop values with home_type as 'Land' WITH bed/bath (Not land)
    merged.drop(merged[((merged['home_type'] == 'Land')) &
                       (merged['bedrooms'] != 0) &
                       (merged['bathrooms'] != 0)
                       ].index, inplace=True)

    # Replace 'Land' values with levels anything other than 0 to 0
    land_repl = {'One Story': 0,
                 'Ground Floor': 0,
                 '2nd Floor': 0,
                 '3rd Floor': 0
                 }
    merged.loc[merged['home_type'] == 'Land', 'levels'] = merged.loc[merged['home_type'] == 'Land', 'levels'].replace(
        land_repl)

    # HOUSE CLEANUP

    # Drop unnecessary houses with irrelevant "levels" for clarity
    house_drop = ['Top Floor', 'Service Floor', 'Attic', 'Other-One',
                  'Split Level', 'Multi-Level', 'Mezzanine', 'Basement',
                  'Basement Level', 'Triplex', 'Duplex', 'Multi/Split',
                  'Three or More Stories-Three Or More', 'Multi/Split-Tri-Level',
                  'Three Or More-Split Level', 'Tri-Level-Two',
                  'Multi/Split-Three Or More', 'One-Multi/Split',
                  'One-Two-Three Or More', 'Two-Three Or More', 'Two Story-One', 'Other'
                  ]
    merged.drop(merged[merged['levels'].isin(house_drop)].index, inplace=True)

    # Convert all 'levels' into numbers
    house_repl = {'One Story': 1, 'Ground Floor': 1, 'One Story': 1, 'Two Story': 1, 'One': 1, 'Two': 2,
                  'Three Or More': 3,
                  'One-Two': 1, 'Three': 3, 'Tri-Level': 3, 'Three Or More-Multi/Split': 3,
                  'Three or More Stories': 3, 'One Story-One': 1, 'Two-Multi/Split': 2,
                  'Two Story-Two': 2, 'Multi/Split-One': 1, 'Multi/Split-Two': 2, 'One-Two-Multi/Split': 1,
                  'One-Three Or More': 1, '4+': 4, 'Tri-Level-Three Or More': 3, 'Three or More Stories-Two': 1,
                  'Three Or More-Two': 1, 'Three or More Stories-One-Two': 2, 'Two Story-Three Or More': 2,
                  'Two-One': 1,
                  '1st Floor': 1, '2nd Floor': 1, '3rd Floor': 1, '0': 1, 0: 1
                  }
    merged.loc[merged['home_type'] == 'House', 'levels'] = merged.loc[merged['home_type'] == 'House', 'levels'].replace(
        house_repl)

    # MULTI-UNIT HOUSING CLEANUP
    # Drop unnecessary units with irrelevant "levels" for clarity
    multi_drop = ['Two-Three Or More-Multi/Split', 'Multi/Split-Two', 'Two-Multi/Split', 'One Story-Two',
                  'Tri-Level-Three Or More', 'Three Or More-Multi/Split', 'Two Story-Three Or More',
                  'Three or More Stories-Two', 'One-Three Or More', 'Two Story'
                  ]
    merged.drop(merged[merged['levels'].isin(multi_drop)].index, inplace=True)

    # Convert all 'levels' into numbers
    multi_repl = {'One Story': 1, 'One': 1, 'Two': 2, '0': 1, 'Three Or More': 3, 'Tri-Level': 3,
                  'Four': 4, 'One Story-Three Or More': 3, 'One Story-One': 1, 'One-Two': 2,
                  'Two Story-Two': 2, 'Three or More Stories-One': 1, 'Five or More': 5,
                  'Ground Floor': 1, '1st Floor': 1, 'Three or More Stories': 1,
                  '2nd Floor': 1, '3rd Floor': 1, '4th Floor': 1, 'Above 10th Floor': 10,
                  '5th Floor': 5, '6th Floor': 6, '9th Floor': 9, '7th Floor': 7, '8th Floor': 8
                  }
    merged.loc[merged['home_type'] == 'Multi-Unit Housing', 'levels'] = merged.loc[
        merged['home_type'] == 'Multi-Unit Housing', 'levels'].replace(multi_repl)

    # Turn all values to numeric #s, making it easier to filter and visualize
    merged['levels'] = pd.to_numeric(merged['levels'])

    # Dropping all data where living_area is 0, therefore invalid
    merged.drop(merged[merged['living_area'] == 0.0].index, inplace=True)

    return merged

'''
FILTERING DATA

Filter into five subsets
'''
def filter(merged):
    # New dataframe sorted by number of bathrooms
    bathroom_sorted = merged.sort_values(by=['bathrooms'])
    # New dataframe sorted by bedrooms
    bedroom_sorted = merged.sort_values(by=['bedrooms'])
    # New dataframe sorted by year_built
    yearbuilt_sorted = merged.sort_values(by=['year_built'])
    # New dataframe sorted by living_area
    livingarea_sorted = merged.sort_values(by=['living_area'])
    # New dataframe sorted by parking_spaces
    parking_sorted = merged.sort_values(by=['parking_spaces'])

    # New dataframe sorted by TotalArea
    totalarea_sorted = merged.sort_values(by=['TotalArea'])
    # Making sure TotalArea is all positive
    totalarea_sorted['TotalArea'] = abs(totalarea_sorted['TotalArea'])

    ## Filtering 1: Comparing Land prices from portugal and california. (box plot?)
    land_prices = totalarea_sorted.loc[
        (totalarea_sorted['frame'] != 'melbourne') & (totalarea_sorted['home_type'] != 'Land'), ['price', 'TotalArea',
                                                                                                 'frame']]
    # Calculates the price of land, per sqft, and reassigns back to 'price'. Now 'price' contains price per sqft
    land_prices['price'] = land_prices['price'] / land_prices['TotalArea']

    ## Filtering 2: Checking the correlation between price and elevator availability in Multi-Unit Housing across all regions (Bar plot?)
    # noelevator contains mean of prices of Multi-Housing Units with no elevators, elevators vice versa
    elev_aval = merged.loc[merged['home_type'] == 'Multi-Unit Housing', ['Elevator', 'price']]
    noelevator = elev_aval.loc[elev_aval['Elevator'] == False, 'price'].mean()
    elevator = elev_aval.loc[elev_aval['Elevator'] == True, 'price'].mean()

    ## Filtering 3: Average price by number of parking spaces for Multi-Unit Housing in Melbourne (plot?)
    melb_parking = parking_sorted.loc[
        (parking_sorted['frame'] == 'melbourne') &
        (parking_sorted['home_type'] != 'House'),
        ['parking_spaces', 'price']
    ]
    # Removing the outlier
    melb_parking.drop(melb_parking[melb_parking['parking_spaces'] == 7].index, inplace=True)

    # Getting average price per number of parking spaces
    line_data = melb_parking.groupby('parking_spaces')['price'].mean().reset_index()

    # Sorting the data by parking_spaces
    line_data.sort_values(by='parking_spaces', inplace=True)

    # Filtering 4: Impact of year_built on price for [2-bedroom / Multi-Unit Housing / based in California] (scatter plot?)
    year_built = merged.loc[(merged['frame'] == 'california') &
                            (merged['bedrooms'] == 2) &
                            (merged['home_type'] == 'Multi-Unit Housing'),
    ['price', 'year_built']]

    # Filtering 5: How many parking spots does Multi-Unit Housing usually have? (Histogram?)
    multi_parking = parking_sorted.loc[parking_sorted['home_type'] == 'Multi-Unit Housing', ['frame', 'parking_spaces']]

    return land_prices, elev_aval, line_data, year_built
'''
PRICE GRAPH OF ALL THREE REGIONS
'''

def priceRegion(data):
    cali = data[data['frame'] == 'california']
    melbourne = data[data['frame'] == 'melbourne']
    portugal = data[data['frame'] == 'portugal']

    avgPriceCalifornia = cali['price'].mean()
    avgPriceMelbourne = melbourne['price'].mean()
    avgPricePortugal = portugal['price'].mean()
    #Define a list of labels for the graph's x column
    locations = ['California', 'Melbourne', 'Portugal']

    #Define a list of average prices for the graph's y column
    avg_prices = [avgPriceCalifornia / 1000000, avgPriceMelbourne / 1000000, avgPricePortugal / 1000000]

    # Create and figure the bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(locations, avg_prices)

    #Add x and y axis labels
    plt.xlabel('Location')
    plt.ylabel('Average Price (Millions)')

    #Title, tighten layout, and show
    plt.title('Average Price Comparison')
    plt.tight_layout()
    plt.show()

'''
PIE CHART OF DATA SIZE BY REGION
'''

def sizePie():
    #Hard- code the sizes of the dataframes for a pre-cleaned representation of the size of all data
    size_cali =  1380171
    size_melbourne = 285180
    size_portugal = 2865575
    # Labels and sizes
    labels = [f'California', f'Melbourne', f'Portugal']
    sizes = [size_cali, size_melbourne, size_portugal]
    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'black'}
    )
    #Title, tighten, show
    plt.title('Proportional Sizes of Dataframes')
    plt.tight_layout()
    plt.show()

'''
STACKED BAR CHART FOR THE COUNT OF EVERY NUMBER OF BEDROOMS BY REGION
'''


def bedroomBar(data):
    cali = data[data['frame'] == 'california']
    melbourne = data[data['frame'] == 'melbourne']
    portugal = data[data['frame'] == 'portugal']
    #Pull the number of times each room count occurs in each dataframe
    count1 = cali['total_rooms'].value_counts()
    count2 = portugal['total_rooms'].value_counts()
    count3 = melbourne['total_rooms'].value_counts()
    #Make a dataframe of the value counts above. Loc due to there being many outliers, and we want a more readable graph.
    unique_values = pd.concat([count1, count2, count3], axis=1).fillna(0).astype(int).loc[0:15]
    #Sort the unique values by their index (the unique numerical values)
    unique_values_sorted = unique_values.sort_index()
    # Plot the sorted stacked bar chart
    unique_values_sorted.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    #Add labels, title, legend, and show
    plt.xlabel('Room Count')
    plt.ylabel('# of Listings with Room Count')
    plt.title('Frequency of Room Counts in Each Dataset')
    plt.legend(['California', 'Portugal', 'Melbourne'])
    plt.tight_layout()
    plt.show()

'''BOX PLOT OF BEDROOM DISTRIBUTION BY REGION'''
def boxBedroom(data):
    cali = data[data['frame'] == 'california']
    melbourne = data[data['frame'] == 'melbourne']
    portugal = data[data['frame'] == 'portugal']
    # Create boxplots for the three datasets and lay them in the same graph
    plt.boxplot(
        [cali['bedrooms'], portugal['bedrooms'], melbourne['bedrooms']],
        labels=['California', 'Portugal', 'Melbourne']
    )
    # Add labels, title, tighten, show
    plt.xlabel('Datasets')
    plt.ylabel('Bedrooms')
    plt.title('Bedrooms in Each Dataset')
    plt.tight_layout()
    plt.show()

'''
KDE CHART OF DISTRIBUTION OF YEAR BUILT
'''

def yearBuiltKDE(data):
    cali = data[data['frame'] == 'california']
    melbourne = data[data['frame'] == 'melbourne']
    portugal = data[data['frame'] == 'portugal']
    #Figure plot
    plt.figure(figsize=(10, 6))
    #Made kde curves for each region's year built column. Also make one for the entire dataset to compare
    cali['year_built'].plot.kde(label='California', linewidth=2)
    portugal['year_built'].plot.kde(label='Portugal', linewidth=2)
    melbourne['year_built'].plot.kde(label='Melbourne', linewidth=2)
    data['year_built'].plot.kde(label='All Data', linewidth=2)
    #Add labels, title, legend, and show
    plt.xlabel('Years')
    plt.ylabel('Density')
    plt.title('Density of Year Built in Each Dataset')
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
VIOLIN PLOT OF PRICE DISTRIBUTION BY ELEVATOR AVILIBILITY
'''
def violinElevator(data):
    land_prices, elev_aval, line_data, year_built = filter(data)
    #Take 2 subsets of the full dataset: elevator available and no elevator available
    no_elevator_data = elev_aval[elev_aval['Elevator'] == False]['price']
    with_elevator_data = elev_aval[elev_aval['Elevator'] == True]['price']
    #Calculate mean and standard deviation for each subset
    no_elevator_mean = no_elevator_data.mean()
    no_elevator_std = no_elevator_data.std()
    with_elevator_mean = with_elevator_data.mean()
    with_elevator_std = with_elevator_data.std()
    #When we first made this graph, we had problems due to there being outliers that made the actual violin much too small. By filtering daat to be within 2 standard deviations of the mean, we can remove these outliers and make the graph more readable.
    filtered_no_elevator_price = no_elevator_data[
        (no_elevator_data >= no_elevator_mean - 2 * no_elevator_std) &
        (no_elevator_data <= no_elevator_mean + 2 * no_elevator_std)
    ]
    filtered_with_elevator_price = with_elevator_data[
        (with_elevator_data >= with_elevator_mean - 2 * with_elevator_std) &
        (with_elevator_data <= with_elevator_mean + 2 * with_elevator_std)
    ]
    # Prepare data for violin plot and make the plot
    bundle = [filtered_no_elevator_price, filtered_with_elevator_price]
    plt.violinplot(bundle, showmeans=True)
    #Add x ticks, labels, and a title
    plt.xticks([1, 2], ['No Elevator', 'With Elevator'])
    plt.xlabel('Elevator Presence')
    plt.ylabel('Price')
    plt.title('Price Distribution Within 2 Standard Deviations of the Mean')
    plt.tight_layout()
    plt.show()

'''
LINE PLOT OF PRICE BY PARKING SPACES
'''

def priceLine(data):
    #get subsets
    land_prices, elev_aval, line_data, year_built = filter(data)
    #Make a line plot of the average price per number of parking spaces
    line_data['price'].plot(kind='line')
    #Add x and y labels, title, show
    plt.xlabel('Parking Spaces')
    plt.ylabel('Average Price')
    plt.title('Increase in Average Price with Number of Parking Spaces')
    plt.show()

'''
LINE PLOT OF PRICE BY YEAR BUILT
'''

def priceYearBuilt(data):
    land_prices, elev_aval, line_data, year_built = filter(data)
    # Group the year built subset by year built and calculate the mean price for each year
    price_by_year = year_built.groupby('year_built')['price'].mean().reset_index()
    # Sort by year to ensure the line plot is chronological
    price_by_year = price_by_year.sort_values('year_built')
    # Make statistics for the average line's plotting
    min_year = price_by_year['year_built'].min()
    max_year = price_by_year['year_built'].max()
    overall_avg_price = year_built['price'].mean()
    # Plot the line graph
    plt.figure(figsize=(10, 6))
    plt.plot(price_by_year['year_built'], price_by_year['price'], marker='o', linestyle='-')
    plt.title('Average House Price by Year Built, Overlaid with Overall Average Price')
    # Make the average line
    plt.hlines(overall_avg_price, min_year, max_year, colors='r', linestyles='--', label='Overall Average Price')
    # Label, grid, and show
    plt.xlabel('Year Built')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.show()