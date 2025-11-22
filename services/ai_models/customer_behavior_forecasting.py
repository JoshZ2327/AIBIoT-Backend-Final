import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class CustomerBehaviorForecastingModel:
    def __init__(self, data, target_column):
        """
        Initialize the forecasting model with raw customer data.
        :param data: A Pandas DataFrame containing customer behavioral data.
        :param target_column: The name of the column to forecast (e.g., 'monthly_spend').
        """
        self.data = data
        self.target_column = target_column
        self.model = None
        self.features = None
        self.results = {}

    def train(self):
        """
        Train a Random Forest model to predict customer behavior.
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        self.features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.results['rmse'] = rmse
        self.results['feature_importance'] = dict(zip(self.features, self.model.feature_importances_))
        return self.results

    def predict_new(self, new_data):
        """
        Predict target values for new incoming customer behavior data.
        :param new_data: A dictionary or list of dictionaries matching the input feature format.
        :return: List of predictions.
        """
        if self.model is None or self.features is None:
            raise Exception("Model has not been trained yet.")

        new_df = pd.DataFrame(new_data)
        new_df = new_df[self.features]
        predictions = self.model.predict(new_df)
        return predictions.tolist()
