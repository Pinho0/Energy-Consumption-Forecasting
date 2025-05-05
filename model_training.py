import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os

SEED = 42

def prepare_dataset(df):
    #Splitting 
    df_full_train, df_test = train_test_split(df, test_size=0.1, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.1111, random_state=SEED)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.energyconsumption.values
    y_val = df_val.energyconsumption.values
    y_test = df_test.energyconsumption.values

    del df_train['energyconsumption']
    del df_val['energyconsumption']
    del df_test['energyconsumption']

    #One-hot encoding
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    test_dicts = df_test.to_dict(orient='records')

    dv.fit(train_dicts)
    X_train = dv.transform(train_dicts)
    X_val = dv.transform(val_dicts)
    X_test = dv.transform(test_dicts)
    
    return X_train, X_test, y_train, y_test, dv


def train_linear_regression(X_train, X_test, y_train, y_test):
    print("-----------------")
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Linear Regression:")
    print(f"Root Mean Squared Error: {rmse}")
    return linear_regression


def train_SVR(X_train, X_test, y_train, y_test):
    print("-----------------")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("SVR:")
    print(f"Root Mean Squared Error: {rmse}")     
    return svr


def train_random_forest_regressor(X_train, X_test, y_train, y_test):
    print("-----------------")
    model = RandomForestRegressor(n_estimators=200, max_depth=11, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Random Forest:")
    print(f"Root Mean Squared Error (No Tuned): {rmse}")
    
    #Tunning
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
    param_grid = {
        'n_estimators': list(range(10, 101, 10)),       
        'max_depth': list(range(1, 11)),               
        'min_samples_leaf': [1, 2, 3, 4, 5]               
    }
    rf = RandomForestRegressor(random_state=SEED)
    grid_search = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            scoring=rmse_scorer,
                            cv=5,
                            n_jobs=-1,          
                            verbose=0)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    
    random_forest = RandomForestRegressor(**best_params, random_state=SEED)     
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error (Tunned): {rmse}")
    return random_forest


def train_gradient_boosting_regressor(X_train, X_test, y_train, y_test):
    print("-----------------")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Gradient Boosting Regressor:")
    print(f"Root Mean Squared Error (No Tuned): {rmse}")
    
    #Tunning
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
    param_grid = {
        'n_estimators': list(range(10, 101, 10)),       
        'max_depth': list(range(1, 10)),               
        'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]             
    }
    gbr = GradientBoostingRegressor(random_state=SEED)
    grid_search = GridSearchCV(estimator=gbr,
                            param_grid=param_grid,
                            scoring=rmse_scorer,
                            cv=5,
                            n_jobs=-1,
                            verbose=0)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    gbr_best = GradientBoostingRegressor(**best_params, random_state=SEED)     
    gbr_best.fit(X_train, y_train)
    y_pred = gbr_best.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error (Tuned): {rmse:.4f}")
    return gbr_best


def save_model(asset, file_name):
    output_file = file_name
    with open(output_file, 'wb') as f_out:
        pickle.dump(asset, f_out)


def main():
    df = pd.read_csv('\energy_consumption_processed.csv')
    X_train, X_test, y_train, y_test, vectorizer = prepare_dataset(df)
    linear_regression_model = train_linear_regression(X_train, X_test, y_train, y_test)
    svr_model = train_SVR(X_train, X_test, y_train, y_test)
    gradient_boosting_regressor_model = train_gradient_boosting_regressor(X_train, X_test, y_train, y_test)
    random_forest_regressor_model = train_random_forest_regressor(X_train, X_test, y_train, y_test)

    os.makedirs('./models', exist_ok=True)
    save_model(vectorizer, './models/dict_vectorizer.bin')
    save_model(linear_regression_model, './models/linear_regression_model.bin')
    save_model(svr_model, './models/svr_model.bin')
    save_model(gradient_boosting_regressor_model, './models/gradient_boosting_regressor_model.bin')
    save_model(random_forest_regressor_model, './models/random_forest_regressor_model.bin')


if __name__ == "__main__":
    main()