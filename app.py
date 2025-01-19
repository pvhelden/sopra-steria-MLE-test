"""
This module defines a FastAPI application for predicting California housing prices.
It includes an HTML form (rendered at the home endpoint) and a /predict endpoint
that accepts JSON input, uses a trained model, and returns the predicted price.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import model_predict

app = FastAPI(title="California Housing Price Prediction")

# Mount static files to serve CSS, JS, or other assets from the 'static' folder
app.mount('/static', StaticFiles(directory='static'), name='static')

# Use Jinja2 for rendering templates located in the 'templates' directory
templates = Jinja2Templates(directory='templates')

class HouseFeatures(BaseModel):
    """
    Pydantic model representing the features for housing price prediction.

    Attributes:
        longitude (float): Geographic coordinate for longitude.
        latitude (float): Geographic coordinate for latitude.
        house_age (float): Median age of the house.
        total_rooms (float): Total number of rooms in the property.
        total_bedrooms (float): Total number of bedrooms in the property.
        population (float): Population of the block.
        households (float): Number of households in the block.
        med_inc (float): Median income of the block.
        ocean_prox (str): Category describing proximity to the ocean.
    """
    longitude: float
    latitude: float
    house_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    med_inc: float
    ocean_prox: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Renders the home page (HTML form) using Jinja2 templates.

    Args:
        request (Request): The FastAPI request object, which includes
            details about the HTTP request.

    Returns:
        HTMLResponse: The rendered 'home.html' template with a form
            to input housing features.
    """
    return templates.TemplateResponse('home.html', {'request': request})

@app.post("/predict", response_class=JSONResponse)
def predict_price(features: HouseFeatures):
    """
    Receives housing features via JSON, uses a trained model to predict
    housing price, and returns the prediction as JSON.

    Args:
        features (HouseFeatures): A Pydantic model that validates
            and contains all necessary housing features.

    Returns:
        JSONResponse: A JSON object containing the 'predicted_price' key
            with the rounded float value of the model's prediction.
    """
    prediction = model_predict.predict(
        features.longitude,
        features.latitude,
        features.house_age,
        features.total_rooms,
        features.total_bedrooms,
        features.population,
        features.households,
        features.med_inc,
        features.ocean_prox
    )
    return JSONResponse({'predicted_price': round(float(prediction), 2)})
