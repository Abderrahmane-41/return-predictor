# 📦 Return Risk Predictor

A machine learning-powered system for predicting e-commerce order returns. Features real-time SHAP explanations to understand exactly why each prediction was made.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-orange)

## ✨ Features

- **ML-Powered Predictions**: LightGBM model with 78.6% accuracy and 87% ROC-AUC
- **Real-Time SHAP Explanations**: Every prediction includes feature contribution analysis
- **Risk Assessment**: Classifies orders as LOW, MEDIUM, or HIGH risk
- **Actionable Insights**: Suggestions for reducing return probability
- **Modern UI**: React frontend with visual contribution bars
- **Fast API**: ~50ms response time including SHAP computation

## 🏗️ Architecture

```
return-predictor/
├── backend/                 # Flask API
│   ├── app.py              # Main Flask application
│   ├── explainer.py        # SHAP + rule-based explanations
│   ├── preprocessing.py    # Feature engineering
│   ├── requirements.txt    # Python dependencies
│   └── artifacts/          # Model files
│       ├── final_model.joblib
│       └── model_metadata.json
│
└── frontend/               # React/Vite UI
    ├── src/
    │   ├── App.jsx         # Main component
    │   └── App.css         # Styles
    └── package.json
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip / conda

### Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at `http://localhost:5173`

## 📡 API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "LightGBM",
  "threshold": 0.35,
  "shap_enabled": true
}
```

### Predict Return Risk

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "season": "winter",
  "product_category": "electronics",
  "Wilaya": "Alger",
  "Courrier": "yalidine",
  "Price": 5000
}
```

**Response:**
```json
{
  "probability": 0.7555,
  "prediction": "HIGH_RISK",
  "threshold_used": 0.35,
  "risky_feature": {
    "feature": "Courrier",
    "value": "yalidine",
    "reason": "This courier has 48.0% return rate",
    "suggestion": "Consider switching to dhd (34.0% return rate)"
  },
  "shap_explanation": {
    "base_value": 0.2283,
    "top_contributors": [
      {"name": "courier_encoded", "shap_contribution": 0.4801, "impact": "increases_risk"},
      {"name": "is_high_performance_courier", "shap_contribution": 0.3122, "impact": "increases_risk"},
      {"name": "is_high_risk_category", "shap_contribution": 0.2978, "impact": "increases_risk"}
    ],
    "sum_of_contributions": 0.8998
  },
  "status": "success"
}
```

### Get Options

```http
GET /options
```

Returns valid values for categorical inputs (seasons, categories, couriers, wilayas).

## 🔬 Understanding SHAP Values

SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction:

| Field | Description |
|-------|-------------|
| `base_value` | Average model prediction across training data |
| `shap_contribution` | How much this feature changes the prediction |
| `impact` | Direction: `increases_risk` or `decreases_risk` |
| `top_contributors` | Top 3 most influential features |

**Interpretation:**
- **Positive SHAP** → Feature pushes prediction toward "return"
- **Negative SHAP** → Feature pushes prediction toward "delivered"
- Larger absolute values = more influential features

## 📊 Model Features

The model uses 9 engineered features:

| Feature | Description |
|---------|-------------|
| `Price_raw` | Order price in DZD |
| `season_encoded` | Seasonal encoding (0-3) |
| `category_encoded` | Product category encoding |
| `wilaya_region_encoded` | Regional encoding of Wilaya |
| `courier_encoded` | Courier service encoding |
| `is_southern` | Binary: Southern region flag |
| `is_high_risk_wilaya` | Binary: High return rate wilaya |
| `is_high_risk_category` | Binary: High return rate category |
| `is_high_performance_courier` | Binary: Low return rate courier |

## 🎨 Frontend Features

- **Date Picker**: Automatically converts to season
- **Probability Gauge**: Visual representation of risk
- **Risk Badge**: Color-coded risk level
- **SHAP Analysis**: Interactive feature contribution display
- **Improvement Suggestions**: Actionable recommendations

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Test ROC-AUC | 0.868 |
| Test PR-AUC | 0.891 |
| Test F1 Score | 0.809 |
| Test Accuracy | 78.6% |
| Optimal Threshold | 0.35 |

## ☁️ Deployment (Render)

### Backend Deployment

1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Set the following:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: (uses Procfile automatically)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHAP_ENABLED` | `true` | Set to `false` to disable SHAP on memory-constrained deployments |

### Memory Optimization (Free Tier)

If you experience worker timeouts on Render's free tier:

```bash
# Option 1: Disable SHAP (saves ~200MB RAM)
SHAP_ENABLED=false

# Option 2: Already configured in Procfile
# --timeout 120 (increased from 30s)
# --workers 1 (single worker)
# --preload (load model once)
```

### Frontend Deployment

1. Create a new **Static Site** on Render
2. Set:
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
3. Add environment variable:
   - `VITE_API_URL`: Your backend URL (e.g., `https://your-backend.onrender.com`)

## 🛠️ Tech Stack

**Backend:**
- Flask (Python web framework)
- LightGBM (Gradient boosting)
- SHAP (Model explanations)
- NumPy, Joblib

**Frontend:**
- React 18
- Vite (Build tool)
- Vanilla CSS

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

Built with ❤️ for smarter e-commerce operations.
