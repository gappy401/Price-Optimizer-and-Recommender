import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# import joblib # Commented out since we are mocking model loading
import json
from datetime import datetime

# ===========================================================================
# MOCK ASSETS & CONFIGURATION
# To make the app runnable without external .pkl files, we mock the model/scaler
# and use a simulated profit function.
# ===========================================================================

# Mock Model Metadata (what the app displays)
MOCK_MODEL_METADATA = {
    'model_type': 'XGBoost Regressor (Simulated)',
    'test_r2': 0.925,
    'test_mae': 850.50,
    'last_trained': datetime.now().strftime('%Y-%m-%d')
}

# Mock Feature Metadata (essential for feature calculation order)
MOCK_FEATURE_METADATA = {
    'all_features': [
        'price', 'competitor_price', 'price_diff_competitor', 'price_ratio_competitor',
        'price_pct_vs_competitor', 'is_premium_vs_competitor', 'price_vs_product_norm',
        'month', 'quarter', 'day_of_week', 'week_of_year', 'is_academic_start',
        'is_summer_slowdown', 'is_year_end', 'is_quarter_end', 'is_weekend',
        'competitor_promotion', 'days_since_promotion', 'competitive_intensity',
        'inventory_level', 'inventory_pct', 'high_inventory_flag',
        'price_inventory_pressure', 'overpriced_overstocked',
        'competitor_promo_high_season', 'premium_in_low_season',
        'price_ma_7d', 'price_ma_30d', 'price_momentum',
        'qty_ma_7d', 'qty_ma_30d', 'demand_trend',
        'product_encoded', 'segment_encoded', 'season_encoded'
    ]
}

# Product configurations (Original data)
PRODUCT_CONFIG = {
    'Centrifuge': {'min_price': 10000, 'max_price': 14000, 'base_price': 12000, 'unit_cost': 4800},
    'PCR_System': {'min_price': 6500, 'max_price': 9500, 'base_price': 8000, 'unit_cost': 3200},
    'Microscope': {'min_price': 13000, 'max_price': 17000, 'base_price': 15000, 'unit_cost': 6000},
    'Pipettes': {'min_price': 250, 'max_price': 400, 'base_price': 300, 'unit_cost': 120},
    'Reagent_Kit': {'min_price': 350, 'max_price': 550, 'base_price': 450, 'unit_cost': 180}
}

SEGMENTS = ['Academic', 'Pharma', 'Biotech', 'Government']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Mock model and scaler objects - set to None as they won't be used for actual prediction
mock_model = None
mock_scaler = None

# ===========================================================================
# PAGE CONFIGURATION
# ===========================================================================

st.set_page_config(
    page_title="Price Optimization Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .recommendation-box {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# LOAD MODEL AND METADATA (MODIFIED FOR MOCKING)
# ===========================================================================

@st.cache_resource
def load_model_assets():
    """Load trained model, scaler, and metadata (MOCKED)"""
    st.warning("⚠️ **Running in Simulated Mode:** Model and scaler files (`.pkl`) were not found. Using a deterministic, heuristic-based profit simulation for visualization.", icon="🚨")
    
    # Return mock objects
    return mock_model, mock_scaler, MOCK_MODEL_METADATA, MOCK_FEATURE_METADATA

model, scaler, model_metadata, feature_metadata = load_model_assets()

# ===========================================================================
# FEATURE ENGINEERING FUNCTIONS
# ===========================================================================

def calculate_features(price, product, segment, competitor_price, inventory, month, 
                       days_since_promo=90, competitor_promo=0):
    """
    Calculate all features needed for prediction
    
    This replicates the feature engineering from training
    """
    
    product_config = PRODUCT_CONFIG[product]
    base_price = product_config['base_price']
    
    # Encode categoricals
    product_encoded = list(PRODUCT_CONFIG.keys()).index(product)
    segment_encoded = SEGMENTS.index(segment)
    
    # Month number (1-12)
    month_num = MONTHS.index(month) + 1
    quarter = (month_num - 1) // 3 + 1
    
    # Day of week (assume mid-week = 2)
    day_of_week = 2
    
    # Price features
    price_diff_competitor = price - competitor_price
    price_ratio_competitor = price / competitor_price if competitor_price > 0 else 1.0
    price_pct_vs_competitor = (price - competitor_price) / competitor_price * 100 if competitor_price > 0 else 0
    is_premium_vs_competitor = 1 if price > competitor_price else 0
    price_vs_product_norm = (price - base_price) / base_price * 100
    
    # Temporal features
    is_academic_start = 1 if month_num in [9, 10] else 0
    is_summer_slowdown = 1 if month_num in [6, 7, 8] else 0
    is_year_end = 1 if month_num == 12 else 0
    is_quarter_end = 1 if month_num in [3, 6, 9, 12] else 0
    is_weekend = 0  # Assume weekday
    
    # Season
    if month_num in [12, 1, 2]:
        season_encoded = 3  # Winter
    elif month_num in [3, 4, 5]:
        season_encoded = 2  # Spring
    elif month_num in [6, 7, 8]:
        season_encoded = 1  # Summer
    else:
        season_encoded = 0  # Fall
    
    # Inventory features
    inventory_pct = inventory / 200 * 100
    high_inventory_flag = 1 if inventory > 150 else 0
    
    # Competitive features
    competitive_intensity = competitor_promo * abs(price_diff_competitor)
    
    # Interaction features
    price_inventory_pressure = (price_vs_product_norm / 100) * (inventory_pct / 100)
    overpriced_overstocked = 1 if (price_vs_product_norm > 0 and high_inventory_flag == 1) else 0
    competitor_promo_high_season = competitor_promo * is_academic_start
    premium_in_low_season = is_premium_vs_competitor * is_summer_slowdown
    
    # Historical features (use product averages as proxy)
    price_ma_7d = price * 0.98  # Assume slightly below current
    price_ma_30d = base_price
    price_momentum = (price - base_price) / base_price * 100
    qty_ma_7d = 85  # Average quantities
    qty_ma_30d = 90
    demand_trend = qty_ma_7d - qty_ma_30d
    
    # Create feature dictionary in same order as training
    features = {
        'price': price,
        'competitor_price': competitor_price,
        'price_diff_competitor': price_diff_competitor,
        'price_ratio_competitor': price_ratio_competitor,
        'price_pct_vs_competitor': price_pct_vs_competitor,
        'is_premium_vs_competitor': is_premium_vs_competitor,
        'price_vs_product_norm': price_vs_product_norm,
        'month': month_num,
        'quarter': quarter,
        'day_of_week': day_of_week,
        'week_of_year': (month_num - 1) * 4 + 2,  # Approximate
        'is_academic_start': is_academic_start,
        'is_summer_slowdown': is_summer_slowdown,
        'is_year_end': is_year_end,
        'is_quarter_end': is_quarter_end,
        'is_weekend': is_weekend,
        'competitor_promotion': competitor_promo,
        'days_since_promotion': days_since_promo,
        'competitive_intensity': competitive_intensity,
        'inventory_level': inventory,
        'inventory_pct': inventory_pct,
        'high_inventory_flag': high_inventory_flag,
        'price_inventory_pressure': price_inventory_pressure,
        'overpriced_overstocked': overpriced_overstocked,
        'competitor_promo_high_season': competitor_promo_high_season,
        'premium_in_low_season': premium_in_low_season,
        'price_ma_7d': price_ma_7d,
        'price_ma_30d': price_ma_30d,
        'price_momentum': price_momentum,
        'qty_ma_7d': qty_ma_7d,
        'qty_ma_30d': qty_ma_30d,
        'demand_trend': demand_trend,
        'product_encoded': product_encoded,
        'segment_encoded': segment_encoded,
        'season_encoded': season_encoded
    }
    
    return features

def predict_profit(features_dict, model, scaler, feature_list):
    """
    SIMULATED Profit Prediction Function
    Calculates profit based on a heuristic (Price - Cost) * Simulated_Demand
    to ensure the optimization logic works visually without a real ML model.
    """
    
    price = features_dict['price']
    
    # Find the product based on product_encoded
    try:
        product_idx = features_dict['product_encoded']
        product_name = list(PRODUCT_CONFIG.keys())[product_idx]
    except (KeyError, IndexError):
        # Fallback if encoding isn't available, though it should be.
        return 0 
    
    unit_cost = PRODUCT_CONFIG[product_name]['unit_cost']
    base_price = PRODUCT_CONFIG[product_name]['base_price']
    
    # --- Demand Simulation Logic ---
    # 1. Base Demand: Decreases with price (elasticity curve)
    # Normalized price difference from base price
    price_norm = (price - base_price) / base_price
    
    # Simple linear demand drop (e.g., -50 units per 10% price increase)
    base_demand = 100 - (price_norm * 500) 
    base_demand = max(10, base_demand) # Min demand floor
    
    # 2. Adjustments based on competitive/seasonal factors (captured by features)
    # Lower price relative to competitor = Higher demand
    comp_adjustment = (features_dict['competitor_price'] - price) / base_price * 100
    
    # Academic start month (Sept/Oct) = Higher demand
    season_adjustment = features_dict['is_academic_start'] * 20
    
    # High inventory (pressure to sell) - slight boost to demand simulation
    inventory_adjustment = features_dict['high_inventory_flag'] * 10
    
    # Apply segment multiplier (e.g., Pharma pays more, Academic less)
    segment_multiplier = 1.0
    if features_dict['segment_encoded'] == 1: # Pharma
        segment_multiplier = 1.2
    elif features_dict['segment_encoded'] == 0: # Academic
        segment_multiplier = 0.8
    
    simulated_demand = (base_demand + comp_adjustment + season_adjustment + inventory_adjustment) * segment_multiplier
    simulated_demand = max(1.0, simulated_demand) # Ensure positive demand
    
    # Profit calculation
    expected_profit = (price - unit_cost) * simulated_demand
    
    # Apply noise for realism
    np.random.seed(42) # Set seed for deterministic noise in the main optimization loop
    noise = np.random.normal(0, expected_profit * 0.005) # 0.5% noise
    final_profit = expected_profit + noise
    
    return final_profit

@st.cache_data
def optimize_price(product, segment, competitor_price, inventory, month, 
                   days_since_promo=90, competitor_promo=0, n_points=100):
    """
    Find optimal price by testing price range
    Returns optimal price, expected profit, and full price curve
    """
    
    product_config = PRODUCT_CONFIG[product]
    min_price = product_config['min_price']
    max_price = product_config['max_price']
    unit_cost = product_config['unit_cost']
    
    # Test price range
    price_range = np.linspace(min_price, max_price, n_points)
    profits = []
    
    for price in price_range:
        features = calculate_features(
            price, product, segment, competitor_price, inventory, month,
            days_since_promo, competitor_promo
        )
        
        # Note: The model, scaler, and feature_metadata['all_features'] are passed but 
        # the predict_profit function is currently mocked to rely only on the features_dict.
        profit = predict_profit(features, model, scaler, feature_metadata['all_features'])
        profits.append(profit)
    
    # Find optimal
    profits = np.array(profits)
    optimal_idx = profits.argmax()
    optimal_price = price_range[optimal_idx]
    optimal_profit = profits[optimal_idx]
    
    # Margin check (ensure minimum 20% margin)
    min_price_for_margin = unit_cost / 0.8  # 20% margin
    if optimal_price < min_price_for_margin:
        optimal_price = min_price_for_margin
        # Recalculate profit at this price
        features = calculate_features(
            optimal_price, product, segment, competitor_price, inventory, month,
            days_since_promo, competitor_promo
        )
        optimal_profit = predict_profit(features, model, scaler, feature_metadata['all_features'])
    
    return optimal_price, optimal_profit, price_range, profits

# ===========================================================================
# MAIN APP
# ===========================================================================

# Header
st.markdown('<div class="main-header">💰 Price Optimization Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/FFFFFF?text=Thermo+Fisher+Analytics", use_container_width=True)
    st.markdown("### 🎯 Navigation")
    
    app_mode = st.radio(
        "Select Mode:",
        ["Single Product Optimization", "Batch Optimization", "Scenario Comparison"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.metric("Model Type", model_metadata['model_type'])
    st.metric("Test R²", f"{model_metadata['test_r2']:.3f}")
    st.metric("Test MAE", f"${model_metadata['test_mae']:,.0f}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This tool uses machine learning (or simulation in this demo) to recommend optimal prices
    that maximize profit based on:
    - Product characteristics
    - Customer segment
    - Market conditions
    - Competitive dynamics
    - Seasonality
    """)

# ===========================================================================
# MODE 1: SINGLE PRODUCT OPTIMIZATION
# ===========================================================================

if app_mode == "Single Product Optimization":
    st.header("🎯 Single Product Optimization")
    st.markdown("Get optimal pricing recommendation for a specific scenario")
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    
    # Initialize session state for rerunning without losing settings
    if 'product' not in st.session_state:
        st.session_state.product = list(PRODUCT_CONFIG.keys())[0]
        st.session_state.segment = SEGMENTS[0]
        st.session_state.month = MONTHS[8]
        st.session_state.competitor_price = PRODUCT_CONFIG[st.session_state.product]['base_price']
        st.session_state.inventory = 100
        st.session_state.days_since_promo = 90
        st.session_state.competitor_promo = False

    def update_session_state(key, value):
        st.session_state[key] = value

    with col1:
        product = st.selectbox("Product", list(PRODUCT_CONFIG.keys()), key='product', on_change=update_session_state, args=('product', st.session_state.product))
        segment = st.selectbox("Customer Segment", SEGMENTS, key='segment', on_change=update_session_state, args=('segment', st.session_state.segment))
        month = st.selectbox("Month", MONTHS, index=8, key='month', on_change=update_session_state, args=('month', st.session_state.month)) 
    
    product_config = PRODUCT_CONFIG[product]

    with col2:
        competitor_price = st.number_input(
            "Competitor Price ($)",
            min_value=int(product_config['min_price'] * 0.8),
            max_value=int(product_config['max_price'] * 1.2),
            value=product_config['base_price'],
            step=100,
            key='competitor_price',
            on_change=update_session_state, args=('competitor_price', st.session_state.competitor_price)
        )
        inventory = st.slider("Inventory Level", 0, 200, 100, key='inventory', on_change=update_session_state, args=('inventory', st.session_state.inventory))
        days_since_promo = st.slider("Days Since Last Promotion", 0, 180, 90, key='days_since_promo', on_change=update_session_state, args=('days_since_promo', st.session_state.days_since_promo))
    
    with col3:
        competitor_promo = st.checkbox("Competitor Running Promotion", key='competitor_promo', on_change=update_session_state, args=('competitor_promo', st.session_state.competitor_promo))
        competitor_promo_val = 1 if competitor_promo else 0
        
        st.markdown("#### Quick Scenarios")
        if st.button("🔥 High Season + Low Inventory", use_container_width=True):
            st.session_state.month = "September"
            st.session_state.inventory = 50
            st.session_state.competitor_promo = False
            st.rerun()
        if st.button("❄️ Low Season + High Inventory", use_container_width=True):
            st.session_state.month = "July"
            st.session_state.inventory = 180
            st.session_state.competitor_promo = True
            st.rerun()
    
    # Run optimization
    if st.button("🚀 Optimize Price", type="primary", use_container_width=True):
        with st.spinner("Calculating optimal price..."):
            optimal_price, optimal_profit, price_range, profits = optimize_price(
                product, segment, competitor_price, inventory, month,
                days_since_promo, competitor_promo_val
            )
            
            # Calculate current price profit
            current_price = product_config['base_price']
            features_current = calculate_features(
                current_price, product, segment, competitor_price, inventory, month,
                days_since_promo, competitor_promo_val
            )
            current_profit = predict_profit(features_current, model, scaler, feature_metadata['all_features'])
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Optimization Results")
            
            # Metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Current Price",
                    f"${current_price:,.0f}",
                    delta=None
                )
            
            with metric_col2:
                price_change_pct = (optimal_price - current_price) / current_price * 100
                st.metric(
                    "Optimal Price",
                    f"${optimal_price:,.0f}",
                    delta=f"{price_change_pct:+.1f}%"
                )
            
            with metric_col3:
                st.metric(
                    "Current Profit",
                    f"${current_profit:,.0f}",
                    delta=None
                )
            
            with metric_col4:
                profit_change_pct = (optimal_profit - current_profit) / current_profit * 100
                st.metric(
                    "Expected Profit",
                    f"${optimal_profit:,.0f}",
                    delta=f"{profit_change_pct:+.1f}%"
                )
            
            # Recommendation box
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>💡 Recommendation</h3>
                <p><strong>Action:</strong> {"Increase" if optimal_price > current_price else "Decrease"} 
                price from ${current_price:,.0f} to ${optimal_price:,.0f} 
                ({abs(price_change_pct):.1f}% {"increase" if optimal_price > current_price else "decrease"})</p>
                
                <p><strong>Expected Impact:</strong> Profit will {"increase" if optimal_profit > current_profit else "decrease"} 
                by ${abs(optimal_profit - current_profit):,.0f} ({abs(profit_change_pct):.1f}%)</p>
                
                <p><strong>vs Competitor:</strong> Your optimal price is 
                {abs((optimal_price - competitor_price) / competitor_price * 100):.1f}% 
                {"higher" if optimal_price > competitor_price else "lower"} than competitor 
                (${competitor_price:,.0f})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Price-Profit curve
                fig_curve = go.Figure()
                
                fig_curve.add_trace(go.Scatter(
                    x=price_range,
                    y=profits,
                    mode='lines',
                    name='Expected Profit',
                    line=dict(color='#667eea', width=3)
                ))
                
                fig_curve.add_trace(go.Scatter(
                    x=[optimal_price],
                    y=[optimal_profit],
                    mode='markers',
                    name='Optimal Price',
                    marker=dict(size=15, color='#28a745', symbol='star')
                ))
                
                fig_curve.add_trace(go.Scatter(
                    x=[current_price],
                    y=[current_profit],
                    mode='markers',
                    name='Current Price',
                    marker=dict(size=12, color='#dc3545')
                ))
                
                fig_curve.update_layout(
                    title="Price vs Profit Curve",
                    xaxis_title="Price ($)",
                    yaxis_title="Expected Profit ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_curve, use_container_width=True)
            
            with viz_col2:
                # Sensitivity analysis
                # Run optimization for scenario prices, but use the same fixed context
                def run_scenario(comp_mult=1.0, inv_change=0, comp_promo_val=competitor_promo_val):
                     return optimize_price(
                        product, segment, competitor_price * comp_mult, 
                        max(0, min(200, inventory + inv_change)), month, days_since_promo, comp_promo_val
                     )[1]

                sensitivities = {
                    'Base Case': optimal_profit,
                    'Comp +10%': run_scenario(comp_mult=1.1),
                    'Comp -10%': run_scenario(comp_mult=0.9),
                    'Inv +50': run_scenario(inv_change=50),
                    'Inv -50': run_scenario(inv_change=-50),
                    'Comp Promo ON': run_scenario(comp_promo_val=1)
                }
                
                fig_sens = go.Figure(go.Bar(
                    x=list(sensitivities.keys()),
                    y=list(sensitivities.values()),
                    marker_color=['#28a745', '#17a2b8', '#17a2b8', '#ffc107', '#ffc107', '#dc3545']
                ))
                
                fig_sens.update_layout(
                    title="Sensitivity Analysis (Profit at Optimal Price)",
                    xaxis_title="Scenario",
                    yaxis_title="Expected Profit ($)"
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)

# ===========================================================================
# MODE 2: BATCH OPTIMIZATION
# ===========================================================================

elif app_mode == "Batch Optimization":
    st.header("📦 Batch Optimization")
    st.markdown("Optimize pricing for multiple products simultaneously")
    
    # Input controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_products = st.multiselect(
            "Select Products",
            list(PRODUCT_CONFIG.keys()),
            default=list(PRODUCT_CONFIG.keys())[:3]
        )
        segment = st.selectbox("Customer Segment", SEGMENTS)
    
    with col2:
        month = st.selectbox("Month", MONTHS, index=8)
        inventory = st.slider("Inventory Level (all products)", 0, 200, 100)
    
    # Run batch optimization
    if st.button("🚀 Optimize All", type="primary", use_container_width=True):
        if not selected_products:
            st.warning("Please select at least one product")
        else:
            with st.spinner("Optimizing prices..."):
                results = []
                
                for product in selected_products:
                    config = PRODUCT_CONFIG[product]
                    # Assume competitor price is 5% higher for batch context
                    competitor_price = config['base_price'] * 1.05 
                    
                    optimal_price, optimal_profit, _, _ = optimize_price(
                        product, segment, competitor_price, inventory, month
                    )
                    
                    # Calculate current profit for comparison
                    current_price = config['base_price']
                    features_current = calculate_features(
                        current_price, product, segment, competitor_price, inventory, month
                    )
                    current_profit = predict_profit(features_current, model, scaler, feature_metadata['all_features'])
                    
                    price_change = (optimal_price - current_price) / current_price * 100
                    profit_change = (optimal_profit - current_profit) / current_profit * 100
                    
                    results.append({
                        'Product': product,
                        'Current Price': f"${current_price:,.0f}",
                        'Optimal Price': f"${optimal_price:,.0f}",
                        'Price Change %': f"{price_change:+.1f}%",
                        'Expected Profit': f"${optimal_profit:,.0f}",
                        'Profit Change %': f"{profit_change:+.1f}%",
                        'Competitor Price': f"${competitor_price:,.0f}"
                    })
                
                # Display results table
                st.subheader("📊 Results")
                results_df = pd.DataFrame(results)
                
                # Highlight profitable changes
                def color_change(val):
                    if isinstance(val, str) and ('+' in val or '-' in val):
                        if '+' in val:
                            return 'background-color: #d4edda; color: #155724'
                        elif '-' in val:
                            return 'background-color: #f8d7da; color: #721c24'
                    return ''
                    
                st.dataframe(
                    results_df.style.applymap(color_change, subset=['Price Change %', 'Profit Change %']), 
                    use_container_width=True, 
                    hide_index=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"pricing_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ===========================================================================
# MODE 3: SCENARIO COMPARISON
# ===========================================================================

else:  # Scenario Comparison
    st.header("⚖️ Scenario Comparison")
    st.markdown("Compare optimal pricing across different market and internal scenarios")
    
    # Product selection
    product = st.selectbox("Product", list(PRODUCT_CONFIG.keys()))
    product_config = PRODUCT_CONFIG[product]
    
    # Define scenarios
    scenarios = {
        'High Season + Pharma (High Price/Profit)': {
            'segment': 'Pharma',
            'month': 'September',
            'inventory': 80,
            'competitor_price': product_config['base_price'] * 1.05,
            'description': 'High demand period, premium segment, low competitor pressure.'
        },
        'Low Season + Academic (Low Price/Profit)': {
            'segment': 'Academic',
            'month': 'July',
            'inventory': 150,
            'competitor_price': product_config['base_price'] * 0.95,
            'description': 'Low demand period, price-sensitive segment, high inventory pressure.'
        },
        'Competitor Promo (Price Match/Defense)': {
            'segment': 'Biotech',
            'month': 'March',
            'inventory': 120,
            'competitor_price': product_config['base_price'] * 0.90,
            'competitor_promo': 1,
            'description': 'Competitor is aggressive, requires defensive pricing strategy.'
        },
        'Year-End Clearance (High Volume/Low Margin)': {
            'segment': 'Government',
            'month': 'December',
            'inventory': 190,
            'competitor_price': product_config['base_price'],
            'description': 'High inventory, end-of-year budget spending, focus on volume.'
        }
    }
    
    # Display scenario descriptions
    st.subheader("Defined Scenarios:")
    for name, params in scenarios.items():
        st.caption(f"**{name}**: Segment: {params['segment']} | Month: {params['month']} | Inventory: {params['inventory']} | Comp Price: ${params['competitor_price']:,.0f} | {params['description']}")
    
    if st.button("🔍 Compare Scenarios", type="primary", use_container_width=True):
        with st.spinner("Analyzing scenarios..."):
            comparison_results = []
            
            for scenario_name, params in scenarios.items():
                optimal_price, optimal_profit, _, _ = optimize_price(
                    product=product,
                    segment=params['segment'],
                    competitor_price=params['competitor_price'],
                    inventory=params['inventory'],
                    month=params['month'],
                    competitor_promo=params.get('competitor_promo', 0)
                )
                
                comparison_results.append({
                    'Scenario': scenario_name,
                    'Optimal Price': optimal_price,
                    'Expected Profit': optimal_profit,
                    'Segment': params['segment'],
                    'Month': params['month'],
                    'Inventory': params['inventory'],
                    'Comp Price': params['competitor_price']
                })
            
            results_df = pd.DataFrame(comparison_results)
            
            # Visualize comparison
            st.subheader("Comparison Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_price = px.bar(
                    results_df,
                    x='Scenario',
                    y='Optimal Price',
                    color='Segment',
                    title='Optimal Price by Scenario',
                    text='Optimal Price'
                )
                fig_price.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_price.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                fig_profit = px.bar(
                    results_df,
                    x='Scenario',
                    y='Expected Profit',
                    color='Segment',
                    title='Expected Profit by Scenario',
                    text='Expected Profit'
                )
                fig_profit.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_profit.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_profit, use_container_width=True)
            
            # Summary insights
            st.subheader("📝 Key Insights")
            
            best_scenario = results_df.loc[results_df['Expected Profit'].idxmax()]
            worst_scenario = results_df.loc[results_df['Expected Profit'].idxmin()]
            
            st.success(f"""
            **Highest Profit Scenario:** {best_scenario['Scenario']} 
            - Price: ${best_scenario['Optimal Price']:,.0f} | Profit: ${best_scenario['Expected Profit']:,.0f}
            """)
            
            st.warning(f"""
            **Lowest Profit Scenario:** {worst_scenario['Scenario']} 
            - Price: ${worst_scenario['Optimal Price']:,.0f} | Profit: ${worst_scenario['Expected Profit']:,.0f}
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 Price Optimization Dashboard | Powered by Machine Learning (Simulated Demo)</p>
    <p style='font-size: 0.8rem;'>Model Accuracy (Simulated): R² = {:.2%} | Last Trained: {}</p>
</div>
""".format(model_metadata['test_r2'], model_metadata['last_trained']), unsafe_allow_html=True)