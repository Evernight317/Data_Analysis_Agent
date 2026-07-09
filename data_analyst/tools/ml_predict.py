"""Machine learning prediction tool for the Data Analysis Agent.

Loads a previously trained model and makes predictions on new data.
"""

from typing import Any

import numpy as np
import pandas as pd
import joblib

from data_analyst.tools.data_loader import get_dataframe


def predict(
    model_path: str,
    data_id: str | None = None,
    input_data: dict[str, Any] | None = None,
    return_proba: bool = False,
) -> dict[str, Any]:
    """Make predictions using a previously trained model.

    Args:
        model_path: Path to the saved model file (.pkl).
        data_id: Reference to previously loaded data for prediction.
        input_data: Single-row input as a dict (alternative to data_id).
        return_proba: Whether to return prediction probabilities (classification only).

    Returns:
        Dictionary with predictions and optional probabilities.
    """
    # Load model
    try:
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        return {"error": f"Model file not found: {model_path}"}
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    # Get input data
    if data_id:
        try:
            df = get_dataframe(data_id)
        except KeyError as e:
            return {"error": str(e)}
    elif input_data:
        df = pd.DataFrame([input_data])
    else:
        return {"error": "Either data_id or input_data must be provided."}

    try:
        # Make predictions
        predictions = pipeline.predict(df)

        result = {
            "model_path": model_path,
            "num_predictions": len(predictions),
            "predictions": predictions.tolist() if len(predictions) > 1 else [float(predictions[0]) if isinstance(predictions[0], (np.floating, float)) else int(predictions[0])],
        }

        # Prediction probabilities (classification only)
        if return_proba:
            try:
                proba = pipeline.predict_proba(df)
                # Get class labels if available
                model = pipeline.named_steps.get("model")
                if hasattr(model, "classes_"):
                    classes = model.classes_.tolist()
                    result["probabilities"] = [
                        {str(classes[i]): round(float(p), 4) for i, p in enumerate(row)}
                        for row in proba
                    ]
                else:
                    result["probabilities"] = proba.tolist()
            except AttributeError:
                result["note"] = "Probability prediction not available for this model type."

        # Summary statistics for predictions
        if len(predictions) > 1:
            result["prediction_summary"] = {
                "mean": round(float(np.mean(predictions)), 4),
                "std": round(float(np.std(predictions)), 4),
                "min": round(float(np.min(predictions)), 4),
                "max": round(float(np.max(predictions)), 4),
            }

        return result

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


def get_tool_def() -> dict:
    """Return the Claude tool definition for predict."""
    return {
        "name": "predict",
        "description": (
            "Make predictions using a previously trained and saved model. "
            "Provide either a data_id (for batch predictions on loaded data) "
            "or input_data (for single-row prediction as a dict). "
            "Optionally return prediction probabilities for classification models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the saved model file (.pkl), as returned by train_model."
                },
                "data_id": {
                    "type": "string",
                    "description": "Reference ID of loaded data to predict on. Use this for batch predictions."
                },
                "input_data": {
                    "type": "object",
                    "description": "Single-row input as a dictionary (column_name: value). Use this for single predictions."
                },
                "return_proba": {
                    "type": "boolean",
                    "description": "Whether to return prediction probabilities (classification models only). Default: false."
                }
            },
            "required": ["model_path"]
        }
    }
