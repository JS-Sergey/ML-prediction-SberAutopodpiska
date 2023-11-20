import dill as dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

import __main__
__main__.pd = pd
__main__.dill = dill


app = FastAPI()

# Load the model and additional coordinate files
with open('data/models/car_rental_service_prediction_model.pkl', 'rb') as file:
    model = dill.load(file)

city_coord_df = pd.read_csv('data/additional_files/city_coord_df.csv')
country_coord_df = pd.read_csv('data/additional_files/country_coord_df.csv')


# Create a class for rows
class Form(BaseModel):
    session_id: Union[str, int, float, None]
    client_id: Union[str, int, float, None]
    visit_date: Union[str, int, float, None]
    visit_time: Union[str, int, float, None]
    visit_number: Union[str, int, float, None]
    utm_source: Union[str, int, float, None]
    utm_medium: Union[str, int, float, None]
    utm_campaign: Union[str, int, float, None]
    utm_adcontent: Union[str, int, float, None]
    utm_keyword: Union[str, int, float, None]
    device_category: Union[str, int, float, None]
    device_os: Union[str, int, float, None]
    device_brand: Union[str, int, float, None]
    device_model: Union[str, int, float, None]
    device_screen_resolution: Union[str, int, float, None]
    device_browser: Union[str, int, float, None]
    geo_country: Union[str, int, float, None]
    geo_city: Union[str, int, float, None]


class Prediction(BaseModel):
    Session_id: Union[str, int, float, None]
    Client_id: Union[str, int, float, None]
    City: Union[str, int, float, None]
    Prediction: Union[str, int, float, None]
    ROC_AUC: Union[str, int, float, None]


# Create decorators and prediction
@app.get('/status')
def status():
    return "It's alive!"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):

    df = pd.DataFrame.from_dict([form.dict()])
    df = df.merge(city_coord_df, how="left", on=['geo_city'])
    df = df.merge(country_coord_df, how="left", on=['geo_country'])

    y = model['model'].predict(df)

    return {
        'Session_id': form.session_id,
        'Client_id': form.client_id,
        'City': form.geo_city,
        'Prediction': y[0],
        'ROC_AUC': model['metadata']['ROC AUC(mean)']
    }
