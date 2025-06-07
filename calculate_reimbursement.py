import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import json
import matplotlib.pyplot as plt
import pickle
import os

# Load historical data for training
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Feature engineering function
def feature_engineering(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Enhance input features based on insights from interviews and PRD.

    Parameters:
        trip_duration_days (int): Number of days spent traveling.
        miles_traveled (float): Total miles traveled.
        total_receipts_amount (float): Total dollar amount of submitted receipts.

    Returns:
        dict: Enhanced feature set.
    """
    # Base features
    features = {
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }

    # Add derived features based on interviews and PRD
    features['miles_per_day'] = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
    features['is_five_day_trip'] = 1 if trip_duration_days == 5 else 0
    features['is_long_trip'] = 1 if trip_duration_days > 7 else 0
    features['receipt_efficiency'] = total_receipts_amount / trip_duration_days if trip_duration_days > 0 else 0
    features['is_high_receipt'] = 1 if total_receipts_amount > 800 else 0
    features['is_low_receipt'] = 1 if total_receipts_amount < 50 else 0
    features['rounding_bonus'] = 1 if total_receipts_amount % 1 in [0.49, 0.99] else 0

    return features

# Prepare training data with feature engineering
X = pd.DataFrame([feature_engineering(
    case['input']['trip_duration_days'],
    case['input']['miles_traveled'],
    case['input']['total_receipts_amount']
) for case in data])
y = pd.Series([case['expected_output'] for case in data])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the pickled model exists and matches the current feature set
model_filename = 'reimbursement_model.pkl'
feature_set_filename = 'feature_set.json'

# Save the current feature set for validation
current_features = list(feature_engineering(1, 1, 1).keys())

# Check if the model and feature set exist
retrain_model = False
if os.path.exists(model_filename) and os.path.exists(feature_set_filename):
    with open(feature_set_filename, 'r') as f:
        saved_features = json.load(f)
    if saved_features != current_features:
        print("Feature set has changed. Retraining the model...")
        retrain_model = True
else:
    print("Model or feature set not found. Training a new model...")
    retrain_model = True

if retrain_model:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    with open(feature_set_filename, 'w') as f:
        json.dump(current_features, f)
    print("Model and feature set saved.")
else:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Predict reimbursement using the trained Random Forest model with enhanced features.

    Parameters:
        trip_duration_days (int): Number of days spent traveling.
        miles_traveled (float): Total miles traveled.
        total_receipts_amount (float): Total dollar amount of submitted receipts.

    Returns:
        float: Predicted reimbursement amount.
    """
    input_features = pd.DataFrame([feature_engineering(trip_duration_days, miles_traveled, total_receipts_amount)])
    predicted_reimbursement = model.predict(input_features)[0]
    return round(predicted_reimbursement, 2)

def train_and_evaluate_model():
    """
    Perform cross-validation, calculate R², and generate plots for model evaluation.
    """
    print("Performing cross-validation on training data...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-validation MAE scores: {-cv_scores}")
    print(f"Average CV MAE: {-cv_scores.mean()}")

    # Generate predictions for the validation set
    predictions = model.predict(X_val)

    # Calculate R²
    r2 = r2_score(y_val, predictions)
    print(f"R² (coefficient of determination): {r2}")

    # Visualize actual vs predicted reimbursement amounts
    plt.figure(figsize=(10, 6))
    plt.scatter(X_val['miles_traveled'], y_val, color='blue', label='Actual', alpha=0.6)
    plt.scatter(X_val['miles_traveled'], predictions, color='red', label='Predicted', alpha=0.6)
    plt.title('Actual vs Predicted Reimbursement Amounts')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Reimbursement Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize actual vs predicted reimbursement amounts with R² line
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, predictions, color='blue', alpha=0.6, label='Actual vs Predicted')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linewidth=2, label='Ideal Fit (R² Line)')
    plt.title(f'Actual vs Predicted Reimbursement Amounts (R² = {r2:.2f})')
    plt.xlabel('Actual Reimbursement Amount')
    plt.ylabel('Predicted Reimbursement Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--train":
        print("Performing cross-validation on training data...")
        train_and_evaluate_model()
        sys.exit(0)

    # Suppress model loading message during prediction
    if len(sys.argv) == 4:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        print("       python3 calculate_reimbursement.py --train")
        sys.exit(1)

    trip_duration_days = int(sys.argv[1])
    miles_traveled = float(sys.argv[2])
    total_receipts_amount = float(sys.argv[3])

    reimbursement = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
    print(reimbursement)
