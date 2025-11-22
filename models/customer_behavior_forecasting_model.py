# models/customer_behavior_forecasting_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class CustomerBehaviorForecastingModel:
    def __init__(self, customer_data):
        """
        Initialize the model with historical customer data.
        Expects a Pandas DataFrame with features and a target column named 'behavior'.
        """
        self.customer_data = customer_data
        self.model = RandomForestClassifier(random_state=42)
        self.report = None
        self.accuracy = None
