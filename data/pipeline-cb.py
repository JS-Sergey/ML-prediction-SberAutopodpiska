import datetime
import dill
import pyarrow
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from catboost import CatBoostClassifier

from data.modules.pipe_functions import datetime_converter, geo_data_filler, category_processor, convert_cat_dateypes, \
    device_os_nan_replacer, create_features, filter_data


def pipeline() -> None:
    """
        Конвейер из 7 шагов
        1 шаг: смена типа даты и времени -->
        2 шаг: обработка гео-данных -->
        3 шаг: обработка категориальных данных -->
        4 шаг: заполнение пропусков -->
        5 шаг: смена типа категориальных фич -->
        6 шаг: создание новых фич -->
        7 шаг: удаление лишних фич

    """

    df = pd.read_parquet('data/dataset/cb_df.parquet')
    X = df.drop('target', axis=1)
    y = df['target']

    preprocessor = Pipeline(steps=[
        ('datetime_converter', FunctionTransformer(datetime_converter)),
        ('geo_data_preprocessor', FunctionTransformer(geo_data_filler)),
        ('categories_optimizer', FunctionTransformer(category_processor)),
        ('os_nan_preprocessor', FunctionTransformer(device_os_nan_replacer)),
        ('str_converter', FunctionTransformer(convert_cat_dateypes)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('filter', FunctionTransformer(filter_data)),
    ])

    # Set the model
    model = CatBoostClassifier(
        random_seed=42,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=2,
        cat_features=[0, 1, 2, 3, 4, 5, 6, 7, 12],
        loss_function='Logloss',
        verbose=True
    )

    # Pipline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # CV
    score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc', verbose=True)

    # Train the model on the whole dataset
    pipe.fit(X, y)

    # Save it to a pickle
    model_filename = 'data/models/car_rental_service_prediction_model.pkl'
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'Name': 'Car Rental Service Prediction model',
                'Author': 'Sergey Jangozyan',
                'Version': 1.1,
                'Date': datetime.datetime.now(),
                'Type': type(pipe.named_steps["classifier"]).__name__,
                'ROC AUC(mean)': score.mean()
            }
        }, file)

    print('Model is saved')
    print(f'model: {type(model).__name__}, ROC-AUC(mean): {score.mean():.4f}, ROC-AUC(std): {score.std():.4f}')


if __name__ == '__main__':
    pipeline()
