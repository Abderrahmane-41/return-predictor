"""
Explainer module for Return Prediction Model.
Identifies the riskiest feature and suggests actionable improvements.
CORRECTED with actual courier and category names from Excel data.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from preprocessing import (
    COURIER_RETURN_RATES,
    CATEGORY_RETURN_RATES,
    HIGH_RISK_WILAYAS,
    HIGH_RISK_CATEGORIES,
    HIGH_PERFORMANCE_COURIERS,
    SOUTHERN_REGIONS,
    WILAYA_TO_REGION,
    BASELINE_RETURN_RATE,
)


# Feature importance ranking (from model training - higher = more important)
FEATURE_IMPORTANCE = {
    "Price_raw": 0.234,
    "courier_encoded": 0.198,
    "is_high_performance_courier": 0.142,
    "category_encoded": 0.125,
    "wilaya_region_encoded": 0.098,
    "is_high_risk_category": 0.076,
    "is_southern": 0.054,
    "is_high_risk_wilaya": 0.041,
    "season_encoded": 0.032,
}

# CORRECTED: Safer alternatives for each courier
# Actual couriers: dhd, kazitour, unknown, yalidine, zr_express
COURIER_ALTERNATIVES = {
    "yalidine": ("dhd", "DHD has lower return rates (~34%)"),
    "kazitour": ("dhd", "DHD has lower return rates (~34%)"),
    "unknown": (
        "dhd",
        "Using a known courier like DHD or ZR Express improves reliability",
    ),
}

# CORRECTED: Category risk advice for actual categories
CATEGORY_RISK_ADVICE = {
    "electronics": "Electronics have high return rates. Ensure accurate product descriptions and photos.",
    "toys_games": "Toys & Games have higher return rates. Consider including sizing guides or age recommendations.",
    "appliances": "Appliances have higher return rates. Ensure specifications are clearly stated.",
}


def get_risky_feature(
    input_data: Dict[str, Any], probability: float
) -> Optional[Dict[str, Any]]:
    """
    Identify the most actionable risky feature and provide suggestions.

    Args:
        input_data: Raw user input (season, product_category, Wilaya, Courrier, Price)
        probability: Predicted return probability

    Returns:
        {
            'feature': 'Courrier',
            'value': 'yalidine',
            'reason': 'This courier has 48% return rate (above baseline 56%)',
            'suggestion': 'Consider switching to dhd (34% return rate)',
            'impact': 'Could reduce return probability by ~15%'
        }
        or None if no actionable risk found
    """
    risks = []

    courier = input_data.get("Courrier", "").lower()
    category = input_data.get("product_category", "").lower()
    wilaya = input_data.get("Wilaya", "")

    # Check courier risk (highest impact, most actionable)
    if courier in COURIER_RETURN_RATES:
        courier_rate = COURIER_RETURN_RATES[courier]
        # Not already using a high-performance courier
        if courier_rate > 40 and courier not in HIGH_PERFORMANCE_COURIERS:
            alt, advice = COURIER_ALTERNATIVES.get(
                courier, ("dhd", "Consider a better-performing courier")
            )
            best_rate = COURIER_RETURN_RATES.get("dhd", 34.0)
            reduction = courier_rate - best_rate
            risks.append(
                {
                    "feature": "Courrier",
                    "value": courier,
                    "reason": f"This courier has {courier_rate:.1f}% return rate",
                    "suggestion": f"Consider switching to {alt} ({best_rate:.1f}% return rate)",
                    "impact": f"Could reduce return risk by ~{reduction:.0f} percentage points",
                    "priority": FEATURE_IMPORTANCE["courier_encoded"]
                    + FEATURE_IMPORTANCE["is_high_performance_courier"],
                }
            )

    # Check category risk
    if category in HIGH_RISK_CATEGORIES:
        cat_rate = CATEGORY_RETURN_RATES.get(category, BASELINE_RETURN_RATE)
        advice = CATEGORY_RISK_ADVICE.get(
            category, "This category has higher return rates."
        )
        risks.append(
            {
                "feature": "product_category",
                "value": category,
                "reason": f"{category.replace('_', ' ').title()} has {cat_rate:.1f}% return rate (above baseline)",
                "suggestion": advice,
                "impact": "Product quality and descriptions are key to reducing returns",
                "priority": FEATURE_IMPORTANCE["category_encoded"]
                + FEATURE_IMPORTANCE["is_high_risk_category"],
            }
        )

    # Check Wilaya risk
    if wilaya in HIGH_RISK_WILAYAS:
        region = WILAYA_TO_REGION.get(wilaya, "Unknown")
        risks.append(
            {
                "feature": "Wilaya",
                "value": wilaya,
                "reason": f"{wilaya} is in {region} region with historically high return rates",
                "suggestion": "Consider additional verification for orders from this region",
                "impact": "Delivery logistics in remote areas contribute to higher returns",
                "priority": FEATURE_IMPORTANCE["wilaya_region_encoded"]
                + FEATURE_IMPORTANCE["is_high_risk_wilaya"],
            }
        )

    # Check southern region
    region = WILAYA_TO_REGION.get(wilaya, "Unknown")
    if region in SOUTHERN_REGIONS and wilaya not in HIGH_RISK_WILAYAS:
        risks.append(
            {
                "feature": "Wilaya",
                "value": wilaya,
                "reason": f"Southern region ({region}) has longer delivery times",
                "suggestion": "Set clear delivery time expectations for customers in this region",
                "impact": "Managing expectations can reduce returns due to delivery delays",
                "priority": FEATURE_IMPORTANCE["is_southern"],
            }
        )

    if not risks:
        return None

    # Return highest priority risk
    risks.sort(key=lambda x: x["priority"], reverse=True)
    best_risk = risks[0]
    del best_risk["priority"]  # Remove internal priority field

    return best_risk


def get_risk_level(probability: float, threshold: float = 0.35) -> str:
    """
    Convert probability to human-readable risk level.

    Args:
        probability: Return probability (0-1)
        threshold: Model's optimal threshold (default 0.35)

    Returns:
        'LOW_RISK', 'MEDIUM_RISK', or 'HIGH_RISK'
    """
    if probability < threshold:
        return "LOW_RISK"
    elif probability < 0.6:
        return "MEDIUM_RISK"
    else:
        return "HIGH_RISK"


def get_risk_summary(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate a summary of all risk factors for the input.

    Returns:
        {
            'positive_factors': ['Using high-performance courier', ...],
            'negative_factors': ['High-risk category', ...]
        }
    """
    positive = []
    negative = []

    courier = input_data.get("Courrier", "").lower()
    category = input_data.get("product_category", "").lower()
    wilaya = input_data.get("Wilaya", "")

    # Courier factors
    if courier in HIGH_PERFORMANCE_COURIERS:
        positive.append(f"Using high-performance courier ({courier})")
    elif courier == "unknown":
        negative.append("Unknown courier increases risk")
    elif courier in ["yalidine", "kazitour"]:
        negative.append(f"Courier {courier} has above-average return rates")

    # Category factors
    if category in HIGH_RISK_CATEGORIES:
        negative.append(f"{category.replace('_', ' ').title()} is a high-risk category")
    elif category in ["furniture", "home_textiles_decor", "accessories"]:
        positive.append(
            f"{category.replace('_', ' ').title()} has below-average return rates"
        )

    # Wilaya factors
    if wilaya in HIGH_RISK_WILAYAS:
        negative.append(f"{wilaya} has historically high return rates")

    region = WILAYA_TO_REGION.get(wilaya, "Unknown")
    if region in SOUTHERN_REGIONS:
        negative.append("Southern region has longer delivery logistics")
    elif region == "Centre":
        positive.append("Central region has reliable delivery infrastructure")

    return {"positive_factors": positive, "negative_factors": negative}


def compute_shap_explanation(
    explainer, features: np.ndarray, feature_names: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Compute real-time SHAP values for a single prediction.

    Args:
        explainer: SHAP TreeExplainer object
        features: Preprocessed feature array of shape (1, n_features)
        feature_names: List of feature names in order

    Returns:
        {
            "base_value": 0.56,
            "features": [
                {
                    "name": "Price_raw",
                    "value": 5000.0,
                    "shap_contribution": 0.12,
                    "impact": "increases_risk"
                },
                ...
            ],
            "top_contributors": [...top 3 by absolute contribution...],
            "sum_of_contributions": 0.15
        }
        or None if computation fails
    """
    try:
        # Compute SHAP values
        shap_values = explainer.shap_values(features)

        # Handle binary classification - extract class 1 (return) values
        # shap_values shape: (n_samples, n_features) for single output
        # or list of 2 arrays for binary classification
        if isinstance(shap_values, list):
            # Binary classification: use class 1 (positive class = return)
            shap_vals = shap_values[1][0]
            base_value = float(explainer.expected_value[1])
        else:
            # Single output or already class 1
            shap_vals = shap_values[0]
            base_value = float(explainer.expected_value)

        # Build feature contribution list
        feature_contributions = []
        for i, (name, shap_val) in enumerate(zip(feature_names, shap_vals)):
            feature_value = float(features[0, i])
            contribution = float(shap_val)

            feature_contributions.append(
                {
                    "name": name,
                    "value": feature_value,
                    "shap_contribution": round(contribution, 4),
                    "impact": (
                        "increases_risk" if contribution > 0 else "decreases_risk"
                    ),
                }
            )

        # Sort by absolute contribution for top contributors
        sorted_features = sorted(
            feature_contributions,
            key=lambda x: abs(x["shap_contribution"]),
            reverse=True,
        )

        # Calculate sum of contributions
        sum_contributions = sum(f["shap_contribution"] for f in feature_contributions)

        return {
            "base_value": round(base_value, 4),
            "features": sorted_features,
            "top_contributors": sorted_features[:3],
            "sum_of_contributions": round(sum_contributions, 4),
        }

    except Exception as e:
        print(f"⚠️ SHAP computation failed: {e}")
        import traceback

        traceback.print_exc()
        return None
