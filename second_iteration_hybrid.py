import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import sys
import math
from enhanced_hybrid_approach import predict_enhanced_hybrid

def create_second_iteration_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Second iteration feature set using the enhanced hybrid as the base predictor.
    """
    features = []
    feature_names = []
    
    # === BASIC FEATURES ===
    features.extend([trip_duration_days, miles_traveled, total_receipts_amount])
    feature_names.extend(['trip_days', 'miles', 'receipts'])
    
    # === DERIVED RATIOS ===
    if trip_duration_days > 0:
        miles_per_day = miles_traveled / trip_duration_days
        spending_per_day = total_receipts_amount / trip_duration_days
    else:
        miles_per_day = 0
        spending_per_day = 0
    
    if miles_traveled > 0:
        receipts_per_mile = total_receipts_amount / miles_traveled
    else:
        receipts_per_mile = 0
    
    features.extend([miles_per_day, spending_per_day, receipts_per_mile])
    feature_names.extend(['miles_per_day', 'spending_per_day', 'receipts_per_mile'])
    
    # === INTERACTION FEATURES ===
    features.append(trip_duration_days * total_receipts_amount)
    feature_names.append('trip_x_receipts')
    
    features.append(miles_traveled * total_receipts_amount / 1000)
    feature_names.append('miles_x_receipts')
    
    features.append(trip_duration_days * miles_per_day)
    feature_names.append('trip_x_efficiency')
    
    # === KEY CATEGORICAL FEATURES (keep the most important ones) ===
    features.append(1 if trip_duration_days == 4 else 0)
    feature_names.append('is_four_day')
    
    features.append(1 if trip_duration_days >= 8 else 0)
    feature_names.append('is_long_trip')
    
    features.append(1 if miles_per_day >= 300 else 0)
    feature_names.append('very_high_efficiency')
    
    features.append(1 if receipts_per_mile >= 10 else 0)
    feature_names.append('very_high_rpm')
    
    features.append(1 if receipts_per_mile < 2 else 0)
    feature_names.append('low_rpm')
    
    # === POLYNOMIAL AND LOG FEATURES (most important ones) ===
    features.append(trip_duration_days ** 2)
    feature_names.append('trip_days_squared')
    
    features.append(math.log1p(total_receipts_amount))
    feature_names.append('log_receipts')
    
    features.append(math.log1p(miles_traveled))
    feature_names.append('log_miles')
    
    # === ENHANCED HYBRID PREDICTION AS BASE FEATURE ===
    enhanced_prediction = predict_enhanced_hybrid(trip_duration_days, miles_traveled, total_receipts_amount)
    features.append(enhanced_prediction)
    feature_names.append('enhanced_hybrid_prediction')
    
    enhanced_ratio = enhanced_prediction / total_receipts_amount if total_receipts_amount > 0 else 0
    features.append(enhanced_ratio)
    feature_names.append('enhanced_hybrid_ratio')
    
    # === NEW FEATURES BASED ON ENHANCED HYBRID INSIGHTS ===
    
    # Specific problem case indicators
    features.append(1 if (trip_duration_days == 4 and receipts_per_mile > 30) else 0)
    feature_names.append('is_extreme_4day_rpm')
    
    features.append(1 if (trip_duration_days == 1 and miles_traveled > 1000) else 0)
    feature_names.append('is_extreme_single_day')
    
    features.append(1 if (trip_duration_days >= 9 and 1300 <= total_receipts_amount <= 1400) else 0)
    feature_names.append('is_9day_medium_receipts')
    
    # Ratio-based features that Enhanced XGBoost found important
    features.append(abs(enhanced_ratio - 0.6))
    feature_names.append('enhanced_deviation_from_60pct')
    
    features.append(enhanced_prediction * trip_duration_days / 1000)
    feature_names.append('enhanced_x_trip_scaled')
    
    # Distance from different ratio targets
    features.append(abs(enhanced_ratio - 1.0))
    feature_names.append('enhanced_deviation_from_100pct')
    
    features.append(abs(enhanced_ratio - 0.3))
    feature_names.append('enhanced_deviation_from_30pct')
    
    # Non-linear combinations
    features.append(enhanced_prediction * receipts_per_mile / 100)
    feature_names.append('enhanced_x_rpm_scaled')
    
    features.append(enhanced_prediction / (miles_traveled + 1))
    feature_names.append('enhanced_per_mile')
    
    # Trip category interactions with enhanced prediction
    if trip_duration_days == 4:
        features.append(enhanced_prediction)
    else:
        features.append(0)
    feature_names.append('enhanced_if_4day')
    
    if trip_duration_days >= 8:
        features.append(enhanced_prediction)
    else:
        features.append(0)
    feature_names.append('enhanced_if_long_trip')
    
    # Error-prone pattern detection
    features.append(1 if (receipts_per_mile > 25 and trip_duration_days <= 4) else 0)
    feature_names.append('high_rpm_short_trip')
    
    features.append(1 if (total_receipts_amount < 100 and trip_duration_days >= 7) else 0)
    feature_names.append('low_receipts_long_trip')
    
    # Enhanced prediction confidence indicators
    if total_receipts_amount > 0:
        enhanced_magnitude = abs(enhanced_prediction - total_receipts_amount * 0.5)
        features.append(enhanced_magnitude)
    else:
        features.append(0)
    feature_names.append('enhanced_magnitude_deviation')
    
    return np.array(features), feature_names


def train_second_iteration_model():
    """Train second iteration XGBoost model."""
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("=== TRAINING SECOND ITERATION XGBOOST MODEL ===\n")
    
    # Prepare features
    X = []
    y = []
    feature_names = None
    
    print("Generating second iteration features...")
    for i, case in enumerate(data):
        if i % 200 == 0:
            print(f"  Processing case {i}/1000...")
            
        features, names = create_second_iteration_features(
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        )
        X.append(features)
        y.append(case['expected_output'])
        if feature_names is None:
            feature_names = names
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTraining with {len(feature_names)} features on {len(X)} cases...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=43)  # Different seed
    
    # Train XGBoost with even more sophisticated parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,         # Slightly reduced depth
        learning_rate=0.03,  # Even lower learning rate
        n_estimators=800,    # More trees
        subsample=0.85,      # Slight adjustment
        colsample_bytree=0.85,
        reg_alpha=0.2,       # More L1 regularization
        reg_lambda=1.5,      # More L2 regularization
        gamma=0.1,           # Min split loss
        random_state=43,
        n_jobs=-1
    )
    
    # Train with early stopping
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_error = np.mean(np.abs(val_pred - y_val))
    val_score = np.sum(np.abs(val_pred - y_val))
    
    print(f"Validation MAE: ${val_error:.2f}")
    print(f"Validation Score: {val_score:.2f}")
    
    # Compare with enhanced hybrid predictions
    enhanced_feature_idx = feature_names.index('enhanced_hybrid_prediction')
    enhanced_predictions = X_val[:, enhanced_feature_idx]
    enhanced_error = np.mean(np.abs(enhanced_predictions - y_val))
    enhanced_score = np.sum(np.abs(enhanced_predictions - y_val))
    
    print(f"Enhanced Hybrid MAE: ${enhanced_error:.2f}")
    print(f"Enhanced Hybrid Score: {enhanced_score:.2f}")
    print(f"Second Iteration improvement: {((enhanced_score - val_score) / enhanced_score * 100):.1f}%")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 TOP 15 MOST IMPORTANT FEATURES:")
    for i, (name, imp) in enumerate(feature_importance[:15]):
        print(f"  {i+1:2d}. {name:30s}: {imp:.4f}")
    
    # Performance comparison
    xgb_better = np.abs(val_pred - y_val) < np.abs(enhanced_predictions - y_val)
    enhanced_better = np.abs(enhanced_predictions - y_val) < np.abs(val_pred - y_val)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"  Second Iteration better: {np.sum(xgb_better)} cases ({np.sum(xgb_better)/len(y_val)*100:.1f}%)")
    print(f"  Enhanced Hybrid better: {np.sum(enhanced_better)} cases ({np.sum(enhanced_better)/len(y_val)*100:.1f}%)")
    
    # Save model
    with open('second_iteration_model.pkl', 'wb') as f:
        pickle.dump((model, feature_names), f)
    
    print(f"\n✅ Second iteration model saved to 'second_iteration_model.pkl'")
    
    return model, feature_names


def predict_second_iteration(trip_duration_days, miles_traveled, total_receipts_amount):
    """Make prediction using second iteration model."""
    
    # Load model
    try:
        with open('second_iteration_model.pkl', 'rb') as f:
            model, feature_names = pickle.load(f)
    except FileNotFoundError:
        print("Second iteration model not found. Please train first.")
        return predict_enhanced_hybrid(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Create features
    features, _ = create_second_iteration_features(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]
    
    return round(prediction, 2)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        model, feature_names = train_second_iteration_model()
    elif len(sys.argv) == 4:
        # Prediction mode
        trip_duration_days = int(float(sys.argv[1]))
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        result = predict_second_iteration(trip_duration_days, miles_traveled, total_receipts_amount)
        print(result)
    else:
        print("Usage:")
        print("  Training: python second_iteration_hybrid.py train")
        print("  Prediction: python second_iteration_hybrid.py <trip_days> <miles> <receipts>")