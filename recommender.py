import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from math import radians, sin, cos, sqrt, atan2
import os

# --- Step 1: Load Data ---
print("Loading data...")
train_customers = pd.read_csv('train_customers.csv')
test_customers = pd.read_csv('test_customers.csv')
train_locations = pd.read_csv('train_locations.csv')
test_locations = pd.read_csv('test_locations.csv')
orders = pd.read_csv('orders.csv', low_memory=False)
vendors = pd.read_csv('vendors.csv')

orders.rename(columns={'LOCATION_NUMBER': 'location_number'}, inplace=True)
vendors.rename(columns={'id': 'vendor_id'}, inplace=True)

# --- Step 2: Feature Aggregates ---
print("Aggregating customer/vendor features...")

customer_agg = orders.groupby('customer_id').agg(
    order_count=('order_id', 'count'),
    avg_total=('grand_total', 'mean'),
    avg_rating=('vendor_rating', 'mean')
).reset_index()

vendor_agg = orders.groupby('vendor_id').agg(
    vendor_order_count=('order_id', 'count'),
    avg_vendor_rating=('vendor_rating', 'mean')
).reset_index()

# --- Step 3: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    except:
        return np.nan

# --- Step 4: Negative Sampling (balanced per customer/location) ---
print("Generating negative samples...")
positive_pairs = orders[['customer_id', 'location_number', 'vendor_id']].drop_duplicates()
negatives = []

for (cust, loc), group in positive_pairs.groupby(['customer_id', 'location_number']):
    all_vendors = set(vendors['vendor_id'])
    pos_vendors = set(group['vendor_id'])
    neg_vendors = list(all_vendors - pos_vendors)
    if neg_vendors:
        sample_size = min(len(pos_vendors) * 2, len(neg_vendors))
        sampled_neg = np.random.choice(neg_vendors, size=sample_size, replace=False)
        for v in sampled_neg:
            negatives.append((cust, loc, v))

negative_df = pd.DataFrame(negatives, columns=['customer_id', 'location_number', 'vendor_id'])
negative_df['target'] = 0

positive_df = positive_pairs.copy()
positive_df['target'] = 1

train_df_raw = pd.concat([positive_df, negative_df], ignore_index=True)

# --- Step 5: Merge Features for Training ---
print("Merging training features...")
train_df = train_df_raw.merge(train_customers, on='customer_id', how='left')
train_df = train_df.merge(train_locations, on=['customer_id', 'location_number'], how='left')
train_df = train_df.merge(vendors, on='vendor_id', how='left')

train_df['cust_latitude'] = train_df['latitude_x']
train_df['cust_longitude'] = train_df['longitude_x']
train_df['vendor_latitude'] = train_df['latitude_y']
train_df['vendor_longitude'] = train_df['longitude_y']

train_df['distance'] = train_df.apply(
    lambda row: haversine(row['cust_latitude'], row['cust_longitude'], 
                          row['vendor_latitude'], row['vendor_longitude']), axis=1)

train_df = train_df.merge(customer_agg, on='customer_id', how='left')
train_df = train_df.merge(vendor_agg, on='vendor_id', how='left')

# Fill missing numeric features
num_cols = ['order_count', 'avg_total', 'avg_rating', 
            'vendor_order_count', 'avg_vendor_rating', 'distance']
for col in num_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)

# --- Step 6: Encode Categoricals ---
categorical_features = []
label_encoders = {}
for col in ['gender', 'location_type', 'vendor_category_en']:
    if col in train_df.columns:
        categorical_features.append(col)
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str)
        train_df[col] = le.fit_transform(train_df[col])
        label_encoders[col] = le

# --- Step 7: Select Features ---
potential_features = ['dob', 'status', 'verified', 'gender', 'location_type',
                      'cust_latitude', 'cust_longitude', 'vendor_latitude', 'vendor_longitude',
                      'distance', 'preparation_time', 'delivery_charge', 'serving_distance',
                      'is_open', 'vendor_rating', 'order_count', 'avg_total', 'avg_rating',
                      'vendor_order_count', 'avg_vendor_rating']

features = [f for f in potential_features if f in train_df.columns]
X = train_df[features]
y = train_df['target']

# --- Step 8: Train Model ---
print("Training LightGBM model...")
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X, y)

# --- Step 9: Prepare Test Data ---
print("Preparing test data...")
test_df = test_locations.merge(vendors, how='cross')
test_df = test_df.merge(test_customers, on='customer_id', how='left')

test_df['cust_latitude'] = test_df['latitude_x']
test_df['cust_longitude'] = test_df['longitude_x']
test_df['vendor_latitude'] = test_df['latitude_y']
test_df['vendor_longitude'] = test_df['longitude_y']

test_df['distance'] = test_df.apply(
    lambda row: haversine(row['cust_latitude'], row['cust_longitude'], 
                          row['vendor_latitude'], row['vendor_longitude']), axis=1)

test_df = test_df.merge(customer_agg, on='customer_id', how='left')
test_df = test_df.merge(vendor_agg, on='vendor_id', how='left')

for col in num_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(0)

for col in categorical_features:
    if col in test_df.columns and col in label_encoders:
        test_df[col] = test_df[col].astype(str)
        # Handle unseen categories by mapping them to the most frequent category
        unseen_mask = ~test_df[col].isin(label_encoders[col].classes_)
        if unseen_mask.any():
            most_frequent = label_encoders[col].classes_[0]  # Use first class as default
            test_df.loc[unseen_mask, col] = most_frequent
        test_df[col] = label_encoders[col].transform(test_df[col])

# Ensure same feature order
X_test = test_df[features]

# --- Step 10: Predict Probabilities ---
print("Predicting probabilities...")
test_df['target'] = lgbm_model.predict_proba(X_test)[:, 1]

# --- Step 11: Get Top 100 Overall Records ---
print("Selecting top 100 records overall...")
top100_df = (
    test_df
    .sort_values('target', ascending=False)
    .head(100)
)

# --- Step 12: Create Submission ---
submission_df = pd.DataFrame({
    'CID X LOC_NUM X VENDOR': top100_df['customer_id'].astype(str) + ' X ' +
                              top100_df['location_number'].astype(str) + ' X ' +
                              top100_df['vendor_id'].astype(str),
    'target': top100_df['target']
})

submission_df = submission_df.sort_values(by=['CID X LOC_NUM X VENDOR'])
submission_df.to_csv('submission_top100.csv', index=False)
print("Submission file saved as 'submission_top100.csv'")