from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
from fastapi.middleware.cors import CORSMiddleware

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

dnn_model = tf.keras.models.load_model('models/DNN_crop_prediction.keras')
logistic_model = joblib.load('models/Logistic Regression.pkl')
naive_bayes_model = joblib.load('models/Naive Bayes Classifier.pkl')
svm_model = joblib.load('models/Support Vector Machine.pkl')
knn_model = joblib.load('models/K-Nearest Neighbors.pkl')
crop_encoder = joblib.load('models/crop_encoder.joblib')
scaler = joblib.load('models/scaler.joblib')


class CropPredictionRequest(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    PH: float
    Rainfall: float


@app.post("/predict")
def predict(data: CropPredictionRequest) -> Dict[str, str]:
    features = np.array(
        [data.Nitrogen, data.Phosphorus, data.Potassium, data.Temperature, data.Humidity, data.PH, data.Rainfall])

    # Calculate the average of the first three values
    average_of_first_three = sum(features[:3]) / 3
    # Create a new list with the average and the remaining values
    modified_test_case_values = np.concatenate((features[3:], np.array([average_of_first_three])))
    print(modified_test_case_values)
    # Convert test case to numpy array and reshape for scaling
    test_case_array = np.array(modified_test_case_values).reshape(1, -1)

    print(test_case_array)

    # Scale the test case using the fitted scaler
    test_case_scaled = scaler.transform(test_case_array)

    logistic_pred = crop_encoder.inverse_transform([logistic_model.predict(test_case_scaled)])[0]
    naive_bayes_pred = crop_encoder.inverse_transform([naive_bayes_model.predict(test_case_scaled)])[0]
    svm_pred = crop_encoder.inverse_transform([svm_model.predict(test_case_scaled)])[0]
    knn_pred = crop_encoder.inverse_transform([knn_model.predict(test_case_scaled)])[0]

    print(test_case_scaled)
    prediction = dnn_model.predict(test_case_scaled)
    crop = np.argmax(prediction[0])
    dnn_pred = crop_encoder.inverse_transform([crop])[0]

    return {
        "DNN": dnn_pred,
        "Logistic Regression": logistic_pred,
        "Naive Bayes": naive_bayes_pred,
        "SVM": svm_pred,
        "K-Nearest Neighbors": knn_pred
    }


@app.get("/accuracy")
def accuracy() -> dict[str, dict[str, float] | dict[str, int] | dict[str, int] | dict[str, int] | dict[str, int]]:
    return {
        "DNN": {"train": 99.15, "test": 96.18},
        "Logistic Regression": {"train": 92, "test": 91.0909},
        "Naive Bayes": {"train": 98.7273, "test": 98.3636},
        "SVM": {"train": 96.9091, "test": 91.6364},
        "K-Nearest Neighbors": {"train": 96.6667, "test": 93.2727},
    }
