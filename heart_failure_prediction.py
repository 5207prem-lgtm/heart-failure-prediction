# Heart Failure Mortality Prediction
# Author: Student Project
# Description: Predicts death event using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    X = data.drop("DEATH_EVENT", axis=1)
    y = data["DEATH_EVENT"]
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=2000))
    ])
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)


def predict_new_patient(model):
    new_patient = [[65, 1, 140, 250000, 1, 30, 140, 1.8, 1, 1, 10, 1, 4]]
    prediction = model.predict(new_patient)

    if prediction[0] == 1:
        print("\nPrediction: High risk of death")
    else:
        print("\nPrediction: Likely to survive")


def main():
    file_path = "heart_failure_clinical_records_dataset.csv"

    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)
    predict_new_patient(model)


if __name__ == "__main__":
    main()
