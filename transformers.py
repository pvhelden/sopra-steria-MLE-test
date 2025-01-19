import pandas as pd

def add_people_per_room(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'people_per_room' column to the DataFrame, computed as population / total_rooms.

    This function copies the input DataFrame to avoid modifying it in place, then
    calculates the ratio of 'population' to 'total_rooms'. Rows where 'total_rooms' is zero
    or missing will result in inf or NaN if not handled beforehand.

    Args:
        X (pd.DataFrame): Input DataFrame containing at least 'population' and 'total_rooms' columns.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'people_per_room' column.
    """
    X = X.copy()
    X['people_per_room'] = X['population'] / X['total_rooms']
    return X

def add_people_per_household(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'people_per_household' column to the DataFrame, computed as population / households.

    This function copies the input DataFrame to avoid modifying it in place, then
    calculates the ratio of 'population' to 'households'. Rows where 'households' is zero
    or missing will result in inf or NaN if not handled beforehand.

    Args:
        X (pd.DataFrame): Input DataFrame containing at least 'population' and 'households' columns.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'people_per_household' column.
    """
    X = X.copy()
    X['people_per_household'] = X['population'] / X['households']
    return X
