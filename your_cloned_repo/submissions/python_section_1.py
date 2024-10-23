from typing import Dict, List
import pandas as pd
from itertools import permutations
import re
import polyline
from pandas import DataFrame


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i + n]
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}

    def flatten(current_dict, parent_key=''):
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    flatten({f"{new_key}[{i}]": item})
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    return flat_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return [list(p) for p in set(permutations(nums))]


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.

    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]

    dates = []

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)

    return dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coords = polyline.decode(polyline_str)

    # Create DataFrame from coordinates
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])

    # Calculate distances using Haversine formula
    df['distance'] = 0.0

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]
        lat2, lon2 = df.iloc[i]

        # Haversine formula to calculate distance
        r = 6371000  # Radius of Earth in meters
        phi1 = lat1 * (3.141592653589793 / 180)
        phi2 = lat2 * (3.141592653589793 / 180)
        delta_phi = (lat2 - lat1) * (3.141592653589793 / 180)
        delta_lambda = (lon2 - lon1) * (3.141592653589793 / 180)

        a = (pow((pd.np.sin(delta_phi / 2)), 2) +
             pd.np.cos(phi1) * pd.np.cos(phi2) *
             pow((pd.np.sin(delta_lambda / 2)), 2))

        c = 2 * pd.np.arctan2(pd.np.sqrt(a), pd.np.sqrt(1 - a))

        distance = r * c

        df.at[i, 'distance'] = distance

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element
    by the sum of its original row and column index before rotation.

    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.

    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)

    # Rotate matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Replace each element with the sum of its original row and column index before rotation
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return final_matrix


def time_check(df) -> DataFrame:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """

    def check_timestamps(group):
        start_times = group['startTime'].min()
        end_times = group['endTime'].max()
        days_covered = set(group['startDay'])

        full_day_covered = start_times <= "00:00" and end_times >= "23:59"
        full_week_covered = len(days_covered) == 7

        return not (full_day_covered and full_week_covered)

    return df.groupby(['id', 'id_2']).apply(check_timestamps).reset_index(drop=True)
