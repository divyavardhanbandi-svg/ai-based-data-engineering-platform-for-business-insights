import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def train_profit_prediction_model(data):
    """Train a simple machine learning model to predict profit."""
    X = data[["Sales", "Quantity", "Discount"]]
    y = data["Profit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)

    return model, error


def predict_profit(model, sales, quantity, discount):
    """Predict profit for new business input values."""
    new_data = pd.DataFrame(
        {
            "Sales": [sales],
            "Quantity": [quantity],
            "Discount": [discount],
        }
    )

    prediction = model.predict(new_data)[0]
    return round(prediction, 2)
