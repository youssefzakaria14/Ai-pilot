
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Federated Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

folder_path = os.path.dirname(os.path.abspath(__file__))

class PatientData(BaseModel):
    Age: int
    Year: int
    Nodes: int

def load_model():
    model_path = os.path.join(folder_path, "federated_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail="Federated model is not available. Run training first."
        )

    return joblib.load(model_path)

@app.get("/")
def home():
    return {
        "success": True,
        "message": "Federated Learning API is working"
    }

@app.get("/dashboard")
def get_dashboard_results():
    metrics_path = os.path.join(folder_path, "metrics_comparison.csv")

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=503,
            detail="Training results are not available. Run training first."
        )

    try:
        metrics_df = pd.read_csv(metrics_path)
        required_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]

        if not all(column in metrics_df.columns for column in required_columns):
            raise HTTPException(
                status_code=500,
                detail="Metrics CSV has an invalid structure"
            )

        evaluations = []

        for _, row in metrics_df.iterrows():
            name = str(row["Model"])

            if "Federated Global Model" in name:
                approach = "federated"
            elif "Centralized Model" in name:
                approach = "centralized"
            elif "(Local)" in name:
                approach = "local"
            else:
                continue

            evaluations.append({
                "model": name,
                "approach": approach,
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1-Score"]),
                "auc": float(row["ROC-AUC"])
            })

        hospital_counts = []
        total_training_rows = 0

        for i in [1, 2, 3]:
            csv_path = os.path.join(folder_path, f"hospital_{i}.csv")

            if not os.path.exists(csv_path):
                raise HTTPException(
                    status_code=503,
                    detail=f"Missing training data: hospital_{i}.csv"
                )

            df = pd.read_csv(csv_path)
            count = len(df)
            total_training_rows += count

            hospital_counts.append({
                "hospitalNumber": i,
                "patientCount": count
            })

        test_path = os.path.join(folder_path, "global_test.csv")

        if not os.path.exists(test_path):
            raise HTTPException(
                status_code=503,
                detail="global_test.csv is missing"
            )

        test_df = pd.read_csv(test_path)

        return {
            "success": True,
            "data": {
                "dataset": "Haberman Dataset",
                "isDemo": False,
                "trainingPatientCount": total_training_rows,
                "testPatientCount": len(test_df),
                "hospitalCounts": hospital_counts,
                "evaluations": evaluations,
                "currentRound": 10,
                "totalRounds": 10,
                "trainingCurve": {
                    "iid": [],
                    "nonIid": []
                },
                "notice": (
                    "Metrics are loaded from the training output CSV. "
                    "Per-round curves and IID/Non-IID comparisons "
                    "are not currently generated."
                )
            }
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dashboard results: {str(error)}"
        )

@app.post("/predict")
def predict(data: PatientData):
    model = load_model()

    input_df = pd.DataFrame(
        [[data.Age, data.Year, data.Nodes]],
        columns=["Age", "Year", "Nodes"]
    )

    try:
        prediction = int(model.predict(input_df.values)[0])
        probabilities = model.predict_proba(input_df.values)[0]
        positive_index = list(model.classes_).index(1)
        probability = round(float(probabilities[positive_index]) * 100, 2)

        return {
            "success": True,
            "prediction": prediction,
            "status": (
                "Survived (>= 5 years)"
                if prediction == 1
                else "High Risk (< 5 years)"
            ),
            "survival_probability": f"{probability}%"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
