import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import sys
import math
from second_iteration_hybrid import predict_second_iteration

def create_third_iteration_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Third iteration feature set using the second iteration as the base predictor.
    """
    features = []
    feature_names = []
    
    # === CORE FEATURES ===
    features.extend([trip_duration_days, miles_traveled, total_receipts_amount])
    feature_names.extend(['trip_days', 'miles', 'receipts'])
    
    # Calculate ratios
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
    
    # === SECOND ITERATION PREDICTION AS PRIMARY FEATURE ===
    second_prediction = predict_second_iteration(trip_duration_days, miles_traveled, total_receipts_amount)
    features.append(second_prediction)
    feature_names.append('second_iteration_prediction')
    
    second_ratio = second_prediction / total_receipts_amount if total_receipts_amount > 0 else 0
    features.append(second_ratio)
    feature_names.append('second_iteration_ratio')
    
    # === FOCUSED HIGH-IMPACT FEATURES ===
    
    # Log features (were important in second iteration)
    features.append(math.log1p(total_receipts_amount))
    feature_names.append('log_receipts')
    
    features.append(math.log1p(miles_traveled))
    feature_names.append('log_miles')
    
    # Key interactions
    features.append(trip_duration_days * total_receipts_amount)
    feature_names.append('trip_x_receipts')
    
    features.append(miles_traveled * total_receipts_amount / 1000)
    feature_names.append('miles_x_receipts')
    
    # Categorical indicators for problem cases
    features.append(1 if trip_duration_days == 4 else 0)
    feature_names.append('is_four_day')
    
    features.append(1 if trip_duration_days >= 8 else 0)
    feature_names.append('is_long_trip')
    
    features.append(1 if receipts_per_mile >= 10 else 0)
    feature_names.append('very_high_rpm')
    
    # === NEW THIRD ITERATION FEATURES ===
    
    # Second iteration deviation patterns
    features.append(abs(second_ratio - 0.6))
    feature_names.append('second_dev_from_60pct')
    
    features.append(abs(second_ratio - 1.0))
    feature_names.append('second_dev_from_100pct')
    
    features.append(abs(second_ratio - 0.3))
    feature_names.append('second_dev_from_30pct')
    
    # Second iteration magnitude
    if total_receipts_amount > 0:
        second_magnitude = abs(second_prediction - total_receipts_amount * 0.5)
        features.append(second_magnitude)
    else:
        features.append(0)
    feature_names.append('second_magnitude_deviation')
    
    # Conditional features based on second iteration performance
    if trip_duration_days >= 8:
        features.append(second_prediction)
    else:
        features.append(0)
    feature_names.append('second_if_long_trip')
    
    if trip_duration_days == 4:
        features.append(second_prediction)
    else:
        features.append(0)
    feature_names.append('second_if_4day')
    
    # Non-linear transformations of second iteration
    features.append(second_prediction * trip_duration_days / 1000)
    feature_names.append('second_x_trip_scaled')
    
    features.append(second_prediction * receipts_per_mile / 100)
    feature_names.append('second_x_rpm_scaled')
    
    # Polynomial features
    features.append(trip_duration_days ** 2)
    feature_names.append('trip_days_squared')
    
    features.append(receipts_per_mile ** 0.5 if receipts_per_mile >= 0 else 0)
    feature_names.append('sqrt_rpm')
    
    # Edge case detection
    features.append(1 if (trip_duration_days == 1 and miles_traveled > 1000) else 0)
    feature_names.append('extreme_single_day')
    
    features.append(1 if (total_receipts_amount < 50) else 0)
    feature_names.append('very_low_receipts')
    
    features.append(1 if (receipts_per_mile > 25 and trip_duration_days <= 4) else 0)
    feature_names.append('extreme_rpm_short')
    
    # Complex ratio combinations
    if miles_traveled > 0 and trip_duration_days > 0:
        efficiency_receipt_ratio = (miles_traveled / trip_duration_days) / (total_receipts_amount / trip_duration_days + 1)
        features.append(efficiency_receipt_ratio)
    else:
        features.append(0)
    feature_names.append('efficiency_receipt_ratio')
    
    return np.array(features), feature_names


def train_third_iteration_model():
    """Train third iteration XGBoost model."""
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("=== TRAINING THIRD ITERATION XGBOOST MODEL ===\n")
    
    # Prepare features
    X = []
    y = []
    feature_names = None
    
    print("Generating third iteration features...")
    for i, case in enumerate(data):
        if i % 200 == 0:
            print(f"  Processing case {i}/1000...")
            
        features, names = create_third_iteration_features(
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
    
    # Split data with different seed
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=44)
    
    # Train XGBoost with refined parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=5,         # Reduced depth for less overfitting
        learning_rate=0.02,  # Very low learning rate
        n_estimators=1000,   # Many trees with low learning rate
        subsample=0.8,       
        colsample_bytree=0.8,
        reg_alpha=0.3,       # High regularization
        reg_lambda=2.0,      
        gamma=0.2,           
        random_state=44,
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
    
    # Compare with second iteration predictions
    second_feature_idx = feature_names.index('second_iteration_prediction')
    second_predictions = X_val[:, second_feature_idx]
    second_error = np.mean(np.abs(second_predictions - y_val))
    second_score = np.sum(np.abs(second_predictions - y_val))
    
    print(f"Second Iteration MAE: ${second_error:.2f}")
    print(f"Second Iteration Score: {second_score:.2f}")
    print(f"Third Iteration improvement: {((second_score - val_score) / second_score * 100):.1f}%")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 TOP 15 MOST IMPORTANT FEATURES:")
    for i, (name, imp) in enumerate(feature_importance[:15]):
        print(f"  {i+1:2d}. {name:30s}: {imp:.4f}")
    
    # Save model
    with open('third_iteration_model.pkl', 'wb') as f:
        pickle.dump((model, feature_names), f)
    
    print(f"\n✅ Third iteration model saved to 'third_iteration_model.pkl'")
    
    return model, feature_names


def predict_third_iteration(trip_duration_days, miles_traveled, total_receipts_amount):
    """Make prediction using third iteration model."""
    
    # Load model
    try:
        with open('third_iteration_model.pkl', 'rb') as f:
            model, feature_names = pickle.load(f)
    except FileNotFoundError:
        print("Third iteration model not found. Please train first.")
        return predict_second_iteration(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Create features
    features, _ = create_third_iteration_features(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]
    
    return round(prediction, 2)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        model, feature_names = train_third_iteration_model()
    elif len(sys.argv) == 4:
        # Prediction mode
        trip_duration_days = int(float(sys.argv[1]))
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        result = predict_third_iteration(trip_duration_days, miles_traveled, total_receipts_amount)
        print(result)
    else:
        print("Usage:")
        print("  Training: python third_iteration_hybrid.py train")
        print("  Prediction: python third_iteration_hybrid.py <trip_days> <miles> <receipts>")