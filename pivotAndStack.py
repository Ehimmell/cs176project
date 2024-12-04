import pandas as pd

def pivotAndStack():
    # 1. Multi-Level Pivot Table with Multiple Aggregations
    data = getData(cali_path, melbourne_path, portugal_path)
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
    print("\nMulti-Level Pivot Table (Aggregated Metrics):")
    print(pivot_table)

    # 2. Stacking the Multi-Level Pivot Table
    stacked_table = pivot_table.stack()
    print("\nStacked Data from Pivot Table:")
    print(stacked_table)

    # 3. Custom Calculations: Price per Bedroom
    data['price_per_bedroom'] = data['price'] / data['bedrooms']
    pivot_price_per_bedroom = data.pivot_table(
        index='city',
        columns='frame',
        values='price_per_bedroom',
        aggfunc='mean'
    )
    print("\nPivot Table (Price Per Bedroom):")
    print(pivot_price_per_bedroom)

    # 4. Merge Stacked Data with Additional Metrics
    # Calculate average price by frame (dataset source)
    avg_price_by_frame = data.groupby('frame')['price'].mean()
    stacked_table_reset = stacked_table.reset_index()  # Reset index for merging
    merged_data = stacked_table_reset.merge(
        avg_price_by_frame, how='left', left_on='frame', right_index=True, suffixes=('', '_avg_frame')
    )
    print("\nMerged Data (Stacked + Average Price by Frame):")
    print(merged_data)