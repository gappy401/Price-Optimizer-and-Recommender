import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate synthetic dataset
def generate_lab_equipment_data(n_records=10000):
    
    # Product categories
    products = {
        'Centrifuge': {'base_price': 12000, 'elasticity': -0.8},
        'PCR_System': {'base_price': 8000, 'elasticity': -1.2},
        'Microscope': {'base_price': 15000, 'elasticity': -0.6},
        'Pipettes': {'base_price': 300, 'elasticity': -1.8},
        'Reagent_Kit': {'base_price': 450, 'elasticity': -1.5}
    }
    
    customer_segments = ['Academic', 'Pharma', 'Biotech', 'Government']
    
    data = []
    start_date = datetime(2021, 1, 1)
    
    for i in range(n_records):
        # Random selections
        product = np.random.choice(list(products.keys()))
        segment = np.random.choice(customer_segments)
        date = start_date + timedelta(days=np.random.randint(0, 1095))  # 3 years
        
        # Base values
        base_price = products[product]['base_price']
        elasticity = products[product]['elasticity']
        
        # Price with variation
        price_variation = np.random.uniform(0.85, 1.15)  # ±15% variation
        actual_price = base_price * price_variation
        
        # Competitor price (correlated but different)
        competitor_price = actual_price * np.random.uniform(0.9, 1.1)
        
        # Segment effects on demand
        segment_multipliers = {
            'Academic': 0.7,   # Lower demand, budget constrained
            'Pharma': 1.5,     # High demand
            'Biotech': 1.2,
            'Government': 1.0
        }
        
        # Seasonality (academic year effect)
        month = date.month
        seasonal_factor = 1.0
        if month in [9, 10]:  # Start of academic year
            seasonal_factor = 1.3
        elif month in [6, 7, 8]:  # Summer slowdown
            seasonal_factor = 0.7
            
        # Calculate quantity demanded
        # Base demand affected by price elasticity, segment, seasonality
        price_effect = (actual_price / base_price) ** elasticity
        base_demand = 100
        
        quantity = int(base_demand * 
                      price_effect * 
                      segment_multipliers[segment] * 
                      seasonal_factor * 
                      np.random.uniform(0.8, 1.2))  # Random noise
        
        quantity = max(0, quantity)  # No negative quantities
        
        # Cost (with economies of scale)
        unit_cost = base_price * 0.4  # 40% COGS
        if quantity > 50:
            unit_cost *= 0.95  # Volume discount
        
        # Inventory level (affects pricing urgency)
        inventory = np.random.randint(0, 200)
        
        # Days since last promotion
        days_since_promo = np.random.randint(0, 180)
        
        # Competitor action (0 = no action, 1 = promotion)
        competitor_promo = np.random.choice([0, 1], p=[0.85, 0.15])
        
        data.append({
            'date': date,
            'product': product,
            'customer_segment': segment,
            'price': round(actual_price, 2),
            'competitor_price': round(competitor_price, 2),
            'quantity_sold': quantity,
            'unit_cost': round(unit_cost, 2),
            'inventory_level': inventory,
            'days_since_promotion': days_since_promo,
            'competitor_promotion': competitor_promo,
            'month': month,
            'quarter': (month - 1) // 3 + 1,
            'day_of_week': date.weekday()
        })
    
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['revenue'] = df['price'] * df['quantity_sold']
    df['cost'] = df['unit_cost'] * df['quantity_sold']
    df['profit'] = df['revenue'] - df['cost']
    df['margin_percent'] = (df['profit'] / df['revenue'] * 100).round(2)
    
    return df

# Generate the dataset
df = generate_lab_equipment_data(10000)
df.to_csv('lab_equipment_pricing.csv', index=False)
print("Dataset generated!")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")