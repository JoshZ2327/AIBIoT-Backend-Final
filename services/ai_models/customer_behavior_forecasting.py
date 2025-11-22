from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class CustomerBehaviorForecastingModel:
    def __init__(self, df, target_column):
        self.df = df
        self.target = target_column
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def preprocess_data(self):
        self.df = pd.get_dummies(self.df)
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        print(f"MAE: {mae}")
        return mae

    def predict(self, new_data):
        return self.model.predict(new_data)
