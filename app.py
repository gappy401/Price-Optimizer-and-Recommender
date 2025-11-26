"""
STREAMLIT APP: PRICE OPTIMIZATION DASHBOARD
===========================================

RUN THIS APP:
    streamlit run app.py

FEATURES:
1. Single product optimization
2. Batch optimization for multiple products
3. Scenario comparison
4. Profit curves visualization
5. Sensitivity analysis

BUSINESS VALUE:
- Instant pricing recommendations
- What-if scenario testing
- Visual profit impact
- Export recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
from datetime import datetime

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
# LOAD MODEL AND METADATA
# ===========================================================================

@st.cache_resource
def load_model_assets():
    """Load trained model, scaler, and metadata"""
    try:
        model = joblib.load('price_optimization_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        with open('model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        
        with open('feature_metadata.json', 'r') as f:
            feature_metadata = json.load(f)
            
        return model, scaler, model_metadata, feature_metadata
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, scaler, model_metadata, feature_metadata = load_model_assets()

# Product configurations
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
    """Predict profit given features"""
    
    # Create DataFrame with features in correct order
    X = pd.DataFrame([features_dict])[feature_list]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    profit_pred = model.predict(X_scaled)[0]
    
    return profit_pred

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
    st.image("https://via.placeholder.com/200x80/667eea/FFFFFF?text=Thermo+Fisher", use_container_width=True)
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
    This tool uses machine learning to recommend optimal prices
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
    
    with col1:
        product = st.selectbox("Product", list(PRODUCT_CONFIG.keys()))
        segment = st.selectbox("Customer Segment", SEGMENTS)
        month = st.selectbox("Month", MONTHS, index=8)  # Default to September
    
    with col2:
        product_config = PRODUCT_CONFIG[product]
        competitor_price = st.number_input(
            "Competitor Price ($)",
            min_value=int(product_config['min_price'] * 0.8),
            max_value=int(product_config['max_price'] * 1.2),
            value=product_config['base_price'],
            step=100
        )
        inventory = st.slider("Inventory Level", 0, 200, 100)
        days_since_promo = st.slider("Days Since Last Promotion", 0, 180, 90)
    
    with col3:
        competitor_promo = st.checkbox("Competitor Running Promotion")
        competitor_promo_val = 1 if competitor_promo else 0
        
        st.markdown("#### Quick Scenarios")
        if st.button("🔥 High Season + Low Inventory"):
            month = "September"
            inventory = 50
            st.experimental_rerun()
        if st.button("❄️ Low Season + High Inventory"):
            month = "July"
            inventory = 180
            st.experimental_rerun()
    
    # Run optimization
    if st.button("🚀 Optimize Price", type="primary"):
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
                sensitivities = {
                    'Base Case': optimal_profit,
                    'Comp +10%': optimize_price(product, segment, competitor_price * 1.1, 
                                                inventory, month, days_since_promo, competitor_promo_val)[1],
                    'Comp -10%': optimize_price(product, segment, competitor_price * 0.9, 
                                                inventory, month, days_since_promo, competitor_promo_val)[1],
                    'Inv +50': optimize_price(product, segment, competitor_price, 
                                              min(200, inventory + 50), month, days_since_promo, competitor_promo_val)[1],
                    'Inv -50': optimize_price(product, segment, competitor_price, 
                                              max(0, inventory - 50), month, days_since_promo, competitor_promo_val)[1],
                }
                
                fig_sens = go.Figure(go.Bar(
                    x=list(sensitivities.keys()),
                    y=list(sensitivities.values()),
                    marker_color=['#28a745', '#17a2b8', '#17a2b8', '#ffc107', '#ffc107']
                ))
                
                fig_sens.update_layout(
                    title="Sensitivity Analysis",
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
    if st.button("🚀 Optimize All", type="primary"):
        if not selected_products:
            st.warning("Please select at least one product")
        else:
            with st.spinner("Optimizing prices..."):
                results = []
                
                for product in selected_products:
                    config = PRODUCT_CONFIG[product]
                    competitor_price = config['base_price'] * 1.05  # Assume 5% higher
                    
                    optimal_price, optimal_profit, _, _ = optimize_price(
                        product, segment, competitor_price, inventory, month
                    )
                    
                    current_price = config['base_price']
                    price_change = (optimal_price - current_price) / current_price * 100
                    
                    results.append({
                        'Product': product,
                        'Current Price': f"${current_price:,.0f}",
                        'Optimal Price': f"${optimal_price:,.0f}",
                        'Change %': f"{price_change:+.1f}%",
                        'Expected Profit': f"${optimal_profit:,.0f}",
                        'Competitor Price': f"${competitor_price:,.0f}"
                    })
                
                # Display results table
                st.subheader("📊 Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"pricing_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ===========================================================================
# MODE 3: SCENARIO COMPARISON
# ===========================================================================

else:  # Scenario Comparison
    st.header("⚖️ Scenario Comparison")
    st.markdown("Compare optimal pricing across different scenarios")
    
    # Product selection
    product = st.selectbox("Product", list(PRODUCT_CONFIG.keys()))
    product_config = PRODUCT_CONFIG[product]
    
    # Define scenarios
    scenarios = {
        'High Season + Pharma': {
            'segment': 'Pharma',
            'month': 'September',
            'inventory': 80,
            'competitor_price': product_config['base_price'] * 1.02
        },
        'Low Season + Academic': {
            'segment': 'Academic',
            'month': 'July',
            'inventory': 150,
            'competitor_price': product_config['base_price'] * 0.95
        },
        'Competitor Promo': {
            'segment': 'Biotech',
            'month': 'March',
            'inventory': 120,
            'competitor_price': product_config['base_price'] * 0.90,
            'competitor_promo': 1
        },
        'High Inventory Clearance': {
            'segment': 'Government',
            'month': 'December',
            'inventory': 190,
            'competitor_price': product_config['base_price']
        }
    }
    
    if st.button("🔍 Compare Scenarios", type="primary"):
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
                    'Month': params['month']
                })
            
            results_df = pd.DataFrame(comparison_results)
            
            # Visualize comparison
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
                st.plotly_chart(fig_profit, use_container_width=True)
            
            # Summary insights
            st.subheader("📝 Key Insights")
            
            best_scenario = results_df.loc[results_df['Expected Profit'].idxmax()]
            worst_scenario = results_df.loc[results_df['Expected Profit'].idxmin()]
            
            st.success(f"""
            **Best Scenario:** {best_scenario['Scenario']}  
            - Optimal Price: ${best_scenario['Optimal Price']:,.0f}  
            - Expected Profit: ${best_scenario['Expected Profit']:,.0f}
            """)
            
            st.warning(f"""
            **Challenging Scenario:** {worst_scenario['Scenario']}  
            - Optimal Price: ${worst_scenario['Optimal Price']:,.0f}  
            - Expected Profit: ${worst_scenario['Expected Profit']:,.0f}
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 Price Optimization Dashboard | Powered by Machine Learning</p>
    <p style='font-size: 0.8rem;'>Model Accuracy: R² = {:.2%} | Last Updated: {}</p>
</div>
""".format(model_metadata['test_r2'], datetime.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)