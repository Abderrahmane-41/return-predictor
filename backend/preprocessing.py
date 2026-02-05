"""
Preprocessing module for Return Prediction Model.
Extracted from Classification.ipynb - converts raw user input to model-ready features.
CORRECTED with actual values from market_sales.xlsx
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# ============================================================================
# STATIC MAPPINGS (Extracted from notebook - ALPHABETICALLY SORTED)
# ============================================================================

# Season encoding (fixed order from notebook)
SEASON_MAPPING = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}

# Wilaya to Region mapping (from notebook WILAYA_REGIONAL_MAP)
WILAYA_TO_REGION = {
    # Centre (8 Wilayas)
    "Alger": "Centre",
    "Blida": "Centre",
    "Boumerdes": "Centre",
    "Tipaza": "Centre",
    "Medea": "Centre",
    "Bouira": "Centre",
    "Tizi Ouzou": "Centre",
    "Ain Defla": "Centre",
    # North East (9 Wilayas)
    "Constantine": "North_East",
    "Annaba": "North_East",
    "Skikda": "North_East",
    "Jijel": "North_East",
    "Bejaia": "North_East",
    "Guelma": "North_East",
    "Mila": "North_East",
    "El Tarf": "North_East",
    "Souk Ahras": "North_East",
    # North West (8 Wilayas)
    "Oran": "North_West",
    "Mostaganem": "North_West",
    "Tlemcen": "North_West",
    "Sidi Bel Abbes": "North_West",
    "Ain Temouchent": "North_West",
    "Mascara": "North_West",
    "Relizane": "North_West",
    "Chlef": "North_West",
    # Interior (14 Wilayas)
    "Setif": "Interior",
    "Batna": "Interior",
    "Djelfa": "Interior",
    "M'Sila": "Interior",
    "Bordj Bou Arreridj": "Interior",
    "Tiaret": "Interior",
    "Tebessa": "Interior",
    "Khenchela": "Interior",
    "Oum El Bouaghi": "Interior",
    "Saida": "Interior",
    "Tissemsilt": "Interior",
    "Naama": "Interior",
    "El Bayadh": "Interior",
    "Laghouat": "Interior",
    # South East (9 Wilayas)
    "Ouargla": "South_East",
    "Biskra": "South_East",
    "El Oued": "South_East",
    "Ghardaia": "South_East",
    "Illizi": "South_East",
    "Touggourt": "South_East",
    "El M'Ghair": "South_East",
    "Ouled Djellal": "South_East",
    "Djanet": "South_East",
    # South West (10 Wilayas)
    "Bechar": "South_West",
    "Adrar": "South_West",
    "Tindouf": "South_West",
    "Tamanrasset": "South_West",
    "Timimoun": "South_West",
    "Beni Abbes": "South_West",
    "In Salah": "South_West",
    "In Guezzam": "South_West",
    "Bordj Badji Mokhtar": "South_West",
    "El Meniaa": "South_West",
    # Unknown
    "Missing_Wilaya": "Unknown",
}

# Region encoding (alphabetical from notebook)
REGION_MAPPING = {
    "Centre": 0,
    "Interior": 1,
    "North_East": 2,
    "North_West": 3,
    "South_East": 4,
    "South_West": 5,
    "Unknown": 6,
}

# ============================================================================
# CORRECTED CATEGORY MAPPING (FROM ACTUAL EXCEL - ALPHABETICALLY SORTED)
# ============================================================================
CATEGORY_MAPPING = {
    "accessories": 0,
    "appliances": 1,
    "beauty": 2,
    "car_parts": 3,
    "electronics": 4,
    "furniture": 5,
    "home_textiles_decor": 6,
    "kitchen": 7,
    "other": 8,
    "perfume": 9,
    "sports_mobility": 10,
    "toys_games": 11,
    "unknown": 12,
}

# ============================================================================
# CORRECTED COURIER MAPPING (FROM ACTUAL EXCEL - ALPHABETICALLY SORTED)
# ============================================================================
COURIER_MAPPING = {
    "dhd": 0,
    "kazitour": 1,
    "unknown": 2,
    "yalidine": 3,
    "zr_express": 4,
}

# ============================================================================
# RISK STATISTICS (From notebook analysis - used for explanations)
# ============================================================================

# Baseline return rate from training data
BASELINE_RETURN_RATE = 55.87  # ~56%

# High-risk thresholds
HIGH_RISK_WILAYA_THRESHOLD = 60.0  # >60% return rate
HIGH_PERFORMANCE_COURIER_THRESHOLD = 35.0  # <35% return rate
HIGH_RISK_CATEGORY_THRESHOLD = BASELINE_RETURN_RATE + 5  # baseline + 5%

# Major cities (Alger, Oran, Constantine)
MAJOR_CITIES = ["Alger", "Oran", "Constantine"]

# Southern regions
SOUTHERN_REGIONS = ["South_East", "South_West"]

# High-risk Wilayas (>60% return rate from training data)
HIGH_RISK_WILAYAS = [
    "Illizi",
    "Tindouf",
    "Tamanrasset",
    "In Guezzam",
    "Bordj Badji Mokhtar",
    "Djanet",
    "In Salah",
    "Beni Abbes",
    "Timimoun",
]

# High-risk categories (>baseline + 5% from training) - UPDATE BASED ON ACTUAL DATA
HIGH_RISK_CATEGORIES = ["electronics", "toys_games"]

# High-performance couriers (<35% return rate) - dhd and zr_express typically
HIGH_PERFORMANCE_COURIERS = ["dhd", "zr_express"]

# Courier return rates (for explanations) - approximate values
COURIER_RETURN_RATES = {
    "zr_express": 32.5,
    "dhd": 34.0,
    "kazitour": 52.3,
    "yalidine": 48.3,
    "unknown": 62.1,
}

# Category return rates (for explanations) - approximate values
CATEGORY_RETURN_RATES = {
    "furniture": 52.1,
    "home_textiles_decor": 53.4,
    "accessories": 54.0,
    "kitchen": 55.2,
    "beauty": 55.5,
    "perfume": 56.0,
    "appliances": 56.5,
    "car_parts": 57.0,
    "sports_mobility": 57.3,
    "other": 58.9,
    "electronics": 62.4,
    "toys_games": 64.2,
    "unknown": 60.0,
}


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================


def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """
    Transform raw user input into 9 features for the model.

    Input format:
        {
            'season': 'winter',
            'product_category': 'electronics',
            'Wilaya': 'Alger',
            'Courrier': 'yalidine',
            'Price': 5000.0
        }

    Output: numpy array of shape (1, 9) with features:
        [Price_raw, season_encoded, category_encoded, wilaya_region_encoded,
         courier_encoded, is_southern, is_high_risk_wilaya,
         is_high_risk_category, is_high_performance_courier]
    """
    # Extract raw values with defaults for unseen values
    season = data.get("season", "winter").lower()
    category = data.get("product_category", "other").lower()
    wilaya = data.get("Wilaya", "Missing_Wilaya")
    courier = data.get("Courrier", "unknown").lower()
    price = float(data.get("Price", 0))

    # Debug prints
    print(
        f"  📥 Raw values: season={season}, category={category}, wilaya={wilaya}, courier={courier}, price={price}"
    )

    # 1. Price_raw (direct copy)
    price_raw = price

    # 2. season_encoded
    season_encoded = SEASON_MAPPING.get(season, SEASON_MAPPING["winter"])

    # 3. category_encoded (handle unseen categories)
    if category not in CATEGORY_MAPPING:
        print(f"  ⚠️  Unknown category '{category}' → using 'other' (8)")
        category_encoded = CATEGORY_MAPPING["other"]
    else:
        category_encoded = CATEGORY_MAPPING[category]

    # 4. wilaya_region_encoded
    region = WILAYA_TO_REGION.get(wilaya, "Unknown")
    wilaya_region_encoded = REGION_MAPPING.get(region, REGION_MAPPING["Unknown"])

    # 5. courier_encoded (handle unseen couriers)
    if courier not in COURIER_MAPPING:
        print(f"  ⚠️  Unknown courier '{courier}' → using 'unknown' (2)")
        courier_encoded = COURIER_MAPPING["unknown"]
    else:
        courier_encoded = COURIER_MAPPING[courier]

    # 6. is_southern (1 if South_East or South_West)
    is_southern = 1 if region in SOUTHERN_REGIONS else 0

    # 7. is_high_risk_wilaya (1 if wilaya return rate > 60%)
    is_high_risk_wilaya = 1 if wilaya in HIGH_RISK_WILAYAS else 0

    # 8. is_high_risk_category (1 if category return rate > baseline + 5%)
    is_high_risk_category = 1 if category in HIGH_RISK_CATEGORIES else 0

    # 9. is_high_performance_courier (1 if courier return rate < 35%)
    is_high_performance_courier = 1 if courier in HIGH_PERFORMANCE_COURIERS else 0

    # Construct feature vector
    features = np.array(
        [
            [
                price_raw,
                season_encoded,
                category_encoded,
                wilaya_region_encoded,
                courier_encoded,
                is_southern,
                is_high_risk_wilaya,
                is_high_risk_category,
                is_high_performance_courier,
            ]
        ],
        dtype=np.float64,
    )

    return features


def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate raw input data and return any warnings.

    Returns:
        (is_valid, warnings_list)
    """
    warnings = []

    # Required fields
    required = ["season", "product_category", "Wilaya", "Courrier", "Price"]
    for field in required:
        if field not in data:
            warnings.append(f"Missing required field: {field}")

    # Validate season
    if "season" in data and data["season"].lower() not in SEASON_MAPPING:
        warnings.append(f"Unknown season '{data['season']}', using 'winter' as default")

    # Validate category
    if (
        "product_category" in data
        and data["product_category"].lower() not in CATEGORY_MAPPING
    ):
        warnings.append(
            f"Unknown category '{data['product_category']}', using 'other' as default"
        )

    # Validate Wilaya
    if "Wilaya" in data and data["Wilaya"] not in WILAYA_TO_REGION:
        warnings.append(
            f"Unknown Wilaya '{data['Wilaya']}', treating as Unknown region"
        )

    # Validate Courier
    if "Courrier" in data and data["Courrier"].lower() not in COURIER_MAPPING:
        warnings.append(
            f"Unknown courier '{data['Courrier']}', using 'unknown' as default"
        )

    # Validate Price
    if "Price" in data:
        try:
            price = float(data["Price"])
            if price < 0:
                warnings.append("Price cannot be negative, using 0")
            if price > 1000000:
                warnings.append(
                    "Price seems unusually high, predictions may be unreliable"
                )
        except (ValueError, TypeError):
            warnings.append(f"Invalid price value '{data['Price']}', using 0")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names used by the model."""
    return [
        "Price_raw",
        "season_encoded",
        "category_encoded",
        "wilaya_region_encoded",
        "courier_encoded",
        "is_southern",
        "is_high_risk_wilaya",
        "is_high_risk_category",
        "is_high_performance_courier",
    ]
