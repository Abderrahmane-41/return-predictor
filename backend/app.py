"""
Flask Backend for Return Prediction API.
Loads the trained LightGBM model and provides prediction + explanation endpoints.
WITH DETAILED DEBUGGING PRINTS
"""

import os
import json
import joblib
import numpy as np
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

from preprocessing import preprocess_input, validate_input, get_feature_names
from explainer import (
    get_risky_feature,
    get_risk_level,
    get_risk_summary,
    compute_shap_explanation,
)


# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Paths to model artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")


# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================


def load_model():
    """Load the trained model and metadata."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}")
        return None


def load_metadata():
    """Load model metadata including optimal threshold."""
    try:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded from {METADATA_PATH}")
        return metadata
    except FileNotFoundError:
        print(f"⚠️ Metadata not found, using defaults")
        return {"optimal_threshold": 0.35, "model_name": "LightGBM"}


def initialize_shap_explainer(model):
    """
    Initialize SHAP TreeExplainer for the loaded model.

    TreeExplainer is exact and fast for tree-based models (LightGBM, XGBoost).
    SHAP values represent feature contributions in model's native space.
    """
    try:
        # TreeExplainer for tree-based models - optimal for LightGBM
        # feature_perturbation='interventional' uses training data distribution
        explainer = shap.TreeExplainer(model)
        print("✅ SHAP TreeExplainer initialized successfully")
        print(f"   Base value (expected value): {explainer.expected_value}")
        return explainer
    except Exception as e:
        print(f"⚠️ SHAP initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return None


# Initialize model and metadata
model = load_model()
metadata = load_metadata()
OPTIMAL_THRESHOLD = metadata.get("optimal_threshold", 0.35)

# Initialize SHAP explainer
shap_explainer = initialize_shap_explainer(model) if model else None


# ============================================================================
# DEBUGGING HELPER
# ============================================================================


def debug_print(title, data, indent=2):
    """Print formatted debug information."""
    prefix = " " * indent
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG: {title}")
    print(f"{'='*60}")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}: {value}")
    elif isinstance(data, (list, np.ndarray)):
        print(f"{prefix}{data}")
    else:
        print(f"{prefix}{data}")
    print(f"{'='*60}\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_name": metadata.get("model_name", "Unknown"),
            "threshold": OPTIMAL_THRESHOLD,
            "shap_enabled": shap_explainer is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint with detailed debugging.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{'#'*80}")
    print(f"# NEW PREDICTION REQUEST - {timestamp}")
    print(f"{'#'*80}")

    if model is None:
        print("❌ ERROR: Model not loaded!")
        return (
            jsonify(
                {"error": "Model not loaded. Check server logs.", "status": "error"}
            ),
            500,
        )

    try:
        # Parse request data
        data = request.get_json()
        if data is None:
            print("❌ ERROR: No JSON in request body")
            return (
                jsonify({"error": "Invalid JSON in request body", "status": "error"}),
                400,
            )

        # DEBUG: Print raw input
        debug_print("RAW INPUT FROM FRONTEND", data)

        # Validate input and collect warnings
        is_valid, warnings = validate_input(data)
        debug_print("VALIDATION RESULT", {"is_valid": is_valid, "warnings": warnings})

        # Preprocess input
        features = preprocess_input(data)
        feature_names = get_feature_names()

        # DEBUG: Print feature vector
        print("\n📊 FEATURE ENGINEERING RESULT:")
        print("-" * 50)
        for i, (name, value) in enumerate(zip(feature_names, features[0])):
            print(f"  [{i}] {name:30s} = {value}")
        print("-" * 50)

        # Get probability prediction
        probability = float(model.predict_proba(features)[0, 1])

        # DEBUG: Print prediction details
        debug_print(
            "MODEL PREDICTION",
            {
                "probability": probability,
                "threshold": OPTIMAL_THRESHOLD,
                "above_threshold": probability >= OPTIMAL_THRESHOLD,
            },
        )

        # Get risk level using optimal threshold
        risk_level = get_risk_level(probability, OPTIMAL_THRESHOLD)

        # DEBUG: Print risk decision
        debug_print(
            "RISK DECISION",
            {
                "probability": f"{probability:.4f} ({probability*100:.2f}%)",
                "threshold": OPTIMAL_THRESHOLD,
                "risk_level": risk_level,
            },
        )

        # Get risky feature explanation
        risky_feature = get_risky_feature(data, probability)

        # DEBUG: Print risky feature
        if risky_feature:
            debug_print("RISKY FEATURE IDENTIFIED", risky_feature)
        else:
            print("ℹ️  No specific risky feature identified")

        # Get risk factors summary
        risk_summary = get_risk_summary(data)
        debug_print("RISK FACTORS SUMMARY", risk_summary)

        # Compute SHAP explanation if explainer is available
        shap_explanation = None
        if shap_explainer is not None:
            shap_explanation = compute_shap_explanation(
                shap_explainer, features, feature_names
            )
            if shap_explanation:
                debug_print(
                    "SHAP EXPLANATION",
                    {
                        "base_value": shap_explanation["base_value"],
                        "top_contributors": shap_explanation["top_contributors"],
                        "sum_of_contributions": shap_explanation[
                            "sum_of_contributions"
                        ],
                    },
                )
        else:
            print("ℹ️  SHAP explainer not available, skipping SHAP computation")

        # Build response
        response = {
            "probability": round(probability, 4),
            "prediction": risk_level,
            "threshold_used": OPTIMAL_THRESHOLD,
            "risky_feature": risky_feature,
            "risk_factors": risk_summary,
            "shap_explanation": shap_explanation,
            "warnings": warnings,
            "status": "success",
        }

        # DEBUG: Print final response
        print("\n✅ FINAL RESPONSE:")
        print("-" * 50)
        print(json.dumps(response, indent=2))
        print("-" * 50)

        return jsonify(response)

    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/features", methods=["GET"])
def get_features():
    """Return the list of features used by the model."""
    return jsonify({"features": get_feature_names(), "total": len(get_feature_names())})


@app.route("/options", methods=["GET"])
def get_options():
    """Return valid options for categorical inputs (for frontend dropdowns)."""
    from preprocessing import (
        SEASON_MAPPING,
        CATEGORY_MAPPING,
        COURIER_MAPPING,
        WILAYA_TO_REGION,
    )

    options = {
        "seasons": list(SEASON_MAPPING.keys()),
        "categories": list(CATEGORY_MAPPING.keys()),
        "couriers": list(COURIER_MAPPING.keys()),
        "wilayas": list(WILAYA_TO_REGION.keys()),
    }

    # DEBUG: Print available options
    debug_print(
        "AVAILABLE OPTIONS",
        {
            "seasons": options["seasons"],
            "categories": options["categories"],
            "couriers": options["couriers"],
            "wilayas_count": len(options["wilayas"]),
        },
    )

    return jsonify(options)


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 RETURN PREDICTION API - DEBUG MODE")
    print("=" * 60)
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Optimal Threshold: {OPTIMAL_THRESHOLD}")
    print(f"Features: {len(get_feature_names())}")

    # Print accepted categories for debugging
    from preprocessing import CATEGORY_MAPPING, COURIER_MAPPING

    print(f"\nAccepted Categories: {list(CATEGORY_MAPPING.keys())}")
    print(f"Accepted Couriers: {list(COURIER_MAPPING.keys())}")
    print("=" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
