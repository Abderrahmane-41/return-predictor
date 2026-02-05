import { useState, useEffect } from 'react'
import './App.css'

// API base URL - change this for production
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

/**
 * Convert a date to a season
 * Spring: March 21 - June 20
 * Summer: June 21 - September 22
 * Autumn: September 23 - December 20
 * Winter: December 21 - March 20
 */
function dateToSeason(dateStr) {
  if (!dateStr) return 'winter'
  
  const date = new Date(dateStr)
  const month = date.getMonth() + 1 // 1-12
  const day = date.getDate()
  
  if ((month === 3 && day >= 21) || month === 4 || month === 5 || (month === 6 && day <= 20)) {
    return 'spring'
  } else if ((month === 6 && day >= 21) || month === 7 || month === 8 || (month === 9 && day <= 22)) {
    return 'summer'
  } else if ((month === 9 && day >= 23) || month === 10 || month === 11 || (month === 12 && day <= 20)) {
    return 'autumn'
  } else {
    return 'winter'
  }
}

/**
 * Get season emoji
 */
function getSeasonEmoji(season) {
  const emojis = {
    spring: '🌸',
    summer: '☀️',
    autumn: '🍂',
    winter: '❄️'
  }
  return emojis[season] || '📅'
}

function App() {
  // Form state - now using date instead of season dropdown
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0], // Today's date
    product_category: 'electronics',
    Wilaya: 'Alger',
    Courrier: 'yalidine',
    Price: 5000
  })
  
  // Computed season from date
  const [computedSeason, setComputedSeason] = useState('winter')
  
  // Result state
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // ============================================================================
  // OPTIONS - MUST MATCH BACKEND preprocessing.py EXACTLY (ALPHABETICALLY SORTED)
  // ============================================================================
  
  // CORRECTED: 13 Categories from actual Excel data (alphabetically sorted)
  const categories = [
    { value: 'accessories', label: 'Accessories' },
    { value: 'appliances', label: 'Appliances' },
    { value: 'beauty', label: 'Beauty' },
    { value: 'car_parts', label: 'Car Parts' },
    { value: 'electronics', label: 'Electronics' },
    { value: 'furniture', label: 'Furniture' },
    { value: 'home_textiles_decor', label: 'Home Textiles & Decor' },
    { value: 'kitchen', label: 'Kitchen' },
    { value: 'other', label: 'Other' },
    { value: 'perfume', label: 'Perfume' },
    { value: 'sports_mobility', label: 'Sports & Mobility' },
    { value: 'toys_games', label: 'Toys & Games' },
    { value: 'unknown', label: 'Unknown' }
  ]
  
  // CORRECTED: 5 Couriers from actual Excel data (alphabetically sorted)
  const couriers = [
    { value: 'dhd', label: 'DHD' },
    { value: 'kazitour', label: 'Kazitour' },
    { value: 'unknown', label: 'Unknown' },
    { value: 'yalidine', label: 'Yalidine' },
    { value: 'zr_express', label: 'ZR Express' }
  ]
  
  // Complete list of Algerian Wilayas
  const wilayas = [
    "Adrar", "Ain Defla", "Ain Temouchent", "Alger", "Annaba", "Batna", "Bechar",
    "Bejaia", "Biskra", "Blida", "Bordj Badji Mokhtar", "Bordj Bou Arreridj",
    "Bouira", "Boumerdes", "Chlef", "Constantine", "Djanet", "Djelfa", "El Bayadh",
    "El M'Ghair", "El Meniaa", "El Oued", "El Tarf", "Ghardaia", "Guelma",
    "Illizi", "In Guezzam", "In Salah", "Jijel", "Khenchela", "Laghouat",
    "M'Sila", "Mascara", "Medea", "Mila", "Mostaganem", "Naama", "Oran",
    "Ouargla", "Ouled Djellal", "Oum El Bouaghi", "Relizane", "Saida", "Setif",
    "Sidi Bel Abbes", "Skikda", "Souk Ahras", "Tamanrasset", "Tebessa",
    "Tiaret", "Timimoun", "Tindouf", "Tipaza", "Tissemsilt", "Tizi Ouzou",
    "Tlemcen", "Touggourt"
  ]
  
  // Update computed season when date changes
  useEffect(() => {
    const season = dateToSeason(formData.date)
    setComputedSeason(season)
    console.log(`📅 Date ${formData.date} → Season: ${season}`)
  }, [formData.date])
  
  // Handle form changes
  const handleChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }))
    
    // Debug log
    console.log(`Form changed: ${name} = ${value}`)
  }
  
  // Submit prediction
  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    
    // Build request payload - convert date to season
    const payload = {
      season: computedSeason,  // Computed from date
      product_category: formData.product_category.toLowerCase(),  // Ensure lowercase
      Wilaya: formData.Wilaya,  // Keep original case for Wilaya
      Courrier: formData.Courrier.toLowerCase(),  // Ensure lowercase
      Price: Number(formData.Price)
    }
    
    console.log('📤 Sending to backend:', payload)
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })
      
      console.log('📥 Response status:', response.status)
      
      const data = await response.json()
      console.log('📥 Response data:', data)
      
      if (data.status === 'error') {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (err) {
      console.error('❌ Fetch error:', err)
      setError(`Connection failed: ${err.message}. Make sure the backend is running on ${API_URL}`)
    } finally {
      setLoading(false)
    }
  }
  
  // Get risk color
  const getRiskColor = (prediction) => {
    switch (prediction) {
      case 'LOW_RISK': return '#10b981'
      case 'MEDIUM_RISK': return '#f59e0b'
      case 'HIGH_RISK': return '#ef4444'
      default: return '#6b7280'
    }
  }
  
  // Get probability color
  const getProbabilityColor = (prob) => {
    if (prob < 0.35) return '#10b981'
    if (prob < 0.6) return '#f59e0b'
    return '#ef4444'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>📦 Return Risk Predictor</h1>
        <p>Predict the likelihood of order returns using machine learning</p>
      </header>
      
      <main className="main">
        <section className="form-section">
          <h2>Order Details</h2>
          <form onSubmit={handleSubmit}>
            {/* Date Input - Converts to Season */}
            <div className="form-group">
              <label htmlFor="date">📅 Order Date</label>
              <input
                type="date"
                id="date"
                name="date"
                value={formData.date}
                onChange={handleChange}
              />
              <div className="season-badge">
                {getSeasonEmoji(computedSeason)} Season: <strong>{computedSeason.charAt(0).toUpperCase() + computedSeason.slice(1)}</strong>
              </div>
            </div>
            
            {/* Product Category */}
            <div className="form-group">
              <label htmlFor="product_category">🏷️ Product Category</label>
              <select
                id="product_category"
                name="product_category"
                value={formData.product_category}
                onChange={handleChange}
              >
                {categories.map(c => (
                  <option key={c.value} value={c.value}>{c.label}</option>
                ))}
              </select>
            </div>
            
            {/* Wilaya */}
            <div className="form-group">
              <label htmlFor="Wilaya">📍 Wilaya (Province)</label>
              <select
                id="Wilaya"
                name="Wilaya"
                value={formData.Wilaya}
                onChange={handleChange}
              >
                {wilayas.map(w => (
                  <option key={w} value={w}>{w}</option>
                ))}
              </select>
            </div>
            
            {/* Courier */}
            <div className="form-group">
              <label htmlFor="Courrier">🚚 Courier</label>
              <select
                id="Courrier"
                name="Courrier"
                value={formData.Courrier}
                onChange={handleChange}
              >
                {couriers.map(c => (
                  <option key={c.value} value={c.value}>{c.label}</option>
                ))}
              </select>
            </div>
            
            {/* Price */}
            <div className="form-group">
              <label htmlFor="Price">💰 Price (DZD)</label>
              <input
                type="number"
                id="Price"
                name="Price"
                value={formData.Price}
                onChange={handleChange}
                min="0"
                step="100"
              />
            </div>
            
            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? '⏳ Analyzing...' : '🔮 Predict Return Risk'}
            </button>
          </form>
          
          {/* Debug Info */}
          <div className="debug-info">
            <details>
              <summary>🔧 Debug: Request Preview</summary>
              <pre>
{JSON.stringify({
  season: computedSeason,
  product_category: formData.product_category,
  Wilaya: formData.Wilaya,
  Courrier: formData.Courrier,
  Price: formData.Price
}, null, 2)}
              </pre>
            </details>
          </div>
        </section>
        
        {/* Results Section */}
        <section className="result-section">
          <h2>Prediction Result</h2>
          
          {error && (
            <div className="error-box">
              <span className="error-icon">❌</span>
              <p>{error}</p>
            </div>
          )}
          
          {result && (
            <div className="result-container">
              {/* Probability Gauge */}
              <div className="probability-gauge">
                <div 
                  className="gauge-fill"
                  style={{
                    width: `${result.probability * 100}%`,
                    backgroundColor: getProbabilityColor(result.probability)
                  }}
                />
                <span className="gauge-label">
                  {(result.probability * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Risk Badge */}
              <div 
                className="risk-badge"
                style={{ backgroundColor: getRiskColor(result.prediction) }}
              >
                {result.prediction === 'LOW_RISK' && '✅ LOW RISK'}
                {result.prediction === 'MEDIUM_RISK' && '⚡ MEDIUM RISK'}
                {result.prediction === 'HIGH_RISK' && '⚠️ HIGH RISK'}
              </div>
              
              {/* Risky Feature Warning */}
              {result.risky_feature && (
                <div className="warning-box">
                  <h3>⚠️ Risk Factor Identified</h3>
                  <p className="warning-feature">
                    <strong>{result.risky_feature.feature}:</strong> {result.risky_feature.value}
                  </p>
                  <p className="warning-reason">{result.risky_feature.reason}</p>
                  <p className="warning-suggestion">
                    💡 <strong>Suggestion:</strong> {result.risky_feature.suggestion}
                  </p>
                  {result.risky_feature.impact && (
                    <p className="warning-impact">📊 {result.risky_feature.impact}</p>
                  )}
                </div>
              )}
              
              {/* Risk Factors Summary */}
              {result.risk_factors && (
                <div className="factors-summary">
                  {result.risk_factors.positive_factors?.length > 0 && (
                    <div className="positive-factors">
                      <h4>✅ Positive Factors</h4>
                      <ul>
                        {result.risk_factors.positive_factors.map((f, i) => (
                          <li key={i}>{f}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {result.risk_factors.negative_factors?.length > 0 && (
                    <div className="negative-factors">
                      <h4>⚠️ Risk Factors</h4>
                      <ul>
                        {result.risk_factors.negative_factors.map((f, i) => (
                          <li key={i}>{f}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              
              {/* Warnings */}
              {result.warnings?.length > 0 && (
                <div className="warnings-list">
                  <h4>⚡ Warnings</h4>
                  <ul>
                    {result.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Debug: Raw Response */}
              <details className="debug-response">
                <summary>🔧 Debug: Raw API Response</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
          
          {!result && !error && !loading && (
            <div className="empty-state">
              <p>Fill in the order details and click "Predict Return Risk" to see results</p>
            </div>
          )}
        </section>
      </main>
      
      <footer className="footer">
        <p>Powered by LightGBM | Model Accuracy: 78.6% | Threshold: 0.35</p>
        <p className="api-info">API: {API_URL}</p>
      </footer>
    </div>
  )
}

export default App
