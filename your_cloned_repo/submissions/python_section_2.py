import pandas as pd
import numpy as np


def calculate_distance_matrix(df) -> pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    distance_matrix = pd.pivot_table(df, values='distance', index='id_start', columns='id_end', fill_value=0)

    # Fill in cumulative distances
    for start in distance_matrix.index:
        for end in distance_matrix.columns:
            if start != end:
                distance_matrix.loc[start, end] = distance_matrix.loc[start, end] + distance_matrix.loc[end, start]

    # Set diagonal to 0
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix


def unroll_distance_matrix(df) -> pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    # Remove self-references
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1

    filtered_ids_df = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    return filtered_ids_df[['id_start']].drop_duplicates()


def calculate_toll_rate(df) -> pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    toll_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in toll_rates.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df) -> pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    df['start_day'] = ''
    df['end_day'] = ''

    # Assuming we want to cover all days and times, we will create a full week schedule
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in days_of_week:
        for hour in range(24):
            start_time = f"{hour:02}:00:00"
            end_time = f"{hour:02}:59:59"
            discount_factor = 0.7 if day in ['Saturday', 'Sunday'] else (0.8 if hour < 10 or hour >= 18 else 1.2)

            # Apply discount factor to each vehicle type based on time of day
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.loc[(df['start_day'] == day), vehicle] *= discount_factor

            # Set start and end days/times
            df['start_day'] = day
            df['end_day'] = day
            df['start_time'] = pd.to_datetime(start_time).time()
            df['end_time'] = pd.to_datetime(end_time).time()

    return df
