import joblib
import pandas as pd

MODEL = joblib.load('housing_price_model.pkl')


def predict(longitude: float, latitude: float, house_age: float, total_rooms: float, total_bedrooms: float,
            population: float, households: float, med_inc: float, ocean_prox: str) -> float:
    """
    Uses a saved pipeline (MODEL) to predict median house value
    for a single data point.

    Args:
        longitude (float): Geographic coordinate (longitude).
        latitude (float): Geographic coordinate (latitude).
        house_age (float): Housing median age.
        total_rooms (float): Total number of rooms in block.
        total_bedrooms (float): Total number of bedrooms in block.
        population (float): Population of the block.
        households (float): Number of households in block.
        med_inc (float): Median income in block.
        ocean_prox (str): Ocean proximity category.

    Returns:
        float: The predicted median house value.
    """
    # Build a single-row DataFrame with the necessary columns
    input_data = pd.DataFrame(
        data=[[
            longitude, latitude, house_age,
            total_rooms, total_bedrooms, population,
            households, med_inc, ocean_prox
        ]],
        columns=[
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', 'ocean_proximity'
        ],
    )

    return MODEL.predict(input_data)[0]
