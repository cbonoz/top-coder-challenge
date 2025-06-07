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

class ReimbursementModel:
    def __init__(self):
        self.model_filename = 'reimbursement_model.pkl'
        self.feature_set_filename = 'feature_set.json'
        self.model = None
        self.current_features = list(self.feature_engineering(1, 1, 1).keys())
        self.X_train, self.X_val, self.y_train, self.y_val, self.X, self.y = self.prepare_training_data()

    def feature_engineering(self, trip_duration_days, miles_traveled, total_receipts_amount):
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

    def prepare_training_data(self):
        """
        Prepare training and validation datasets with feature engineering.

        Returns:
            tuple: X_train, X_val, y_train, y_val, X, y
        """
        # Prepare training data with feature engineering
        X = pd.DataFrame([self.feature_engineering(
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ) for case in data])
        y = pd.Series([case['expected_output'] for case in data])

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val, X, y

    def load_or_train_model(self):
        """
        Load the model from file or train a new one if necessary.
        """
        retrain_model = False
        if os.path.exists(self.model_filename) and os.path.exists(self.feature_set_filename):
            with open(self.feature_set_filename, 'r') as f:
                saved_features = json.load(f)
            if saved_features != self.current_features:
                print("Feature set has changed. Retraining the model...")
                retrain_model = True
        else:
            print("Model or feature set not found. Training a new model...")
            retrain_model = True

        if retrain_model:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            with open(self.model_filename, 'wb') as file:
                pickle.dump(self.model, file)
            with open(self.feature_set_filename, 'w') as f:
                json.dump(self.current_features, f)
            print("Model and feature set saved.")
        else:
            with open(self.model_filename, 'rb') as file:
                self.model = pickle.load(file)

    def calculate_reimbursement(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Predict reimbursement using the trained Random Forest model with enhanced features.

        Parameters:
            trip_duration_days (int): Number of days spent traveling.
            miles_traveled (float): Total miles traveled.
            total_receipts_amount (float): Total dollar amount of submitted receipts.

        Returns:
            float: Predicted reimbursement amount.
        """
        input_features = pd.DataFrame([self.feature_engineering(trip_duration_days, miles_traveled, total_receipts_amount)])
        predicted_reimbursement = self.model.predict(input_features)[0]
        return round(predicted_reimbursement, 2)

    def train_and_evaluate_model(self):
        """
        Perform cross-validation, calculate R², and generate plots for model evaluation.
        """
        print("Performing cross-validation on training data...")
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='neg_mean_absolute_error')
        print(f"Cross-validation MAE scores: {-cv_scores}")
        print(f"Average CV MAE: {-cv_scores.mean()}")

        # Generate predictions for the validation set
        predictions = self.model.predict(self.X_val)

        # Calculate R²
        r2 = r2_score(self.y_val, predictions)
        print(f"R² (coefficient of determination): {r2}")

        # Visualize actual vs predicted reimbursement amounts
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_val['miles_traveled'], self.y_val, color='blue', label='Actual', alpha=0.6)
        plt.scatter(self.X_val['miles_traveled'], predictions, color='red', label='Predicted', alpha=0.6)
        plt.title('Actual vs Predicted Reimbursement Amounts')
        plt.xlabel('Miles Traveled')
        plt.ylabel('Reimbursement Amount')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Visualize actual vs predicted reimbursement amounts with R² line
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_val, predictions, color='blue', alpha=0.6, label='Actual vs Predicted')
        plt.plot([self.y_val.min(), self.y_val.max()], [self.y_val.min(), self.y_val.max()], color='red', linewidth=2, label='Ideal Fit (R² Line)')
        plt.title(f'Actual vs Predicted Reimbursement Amounts (R² = {r2:.2f})')
        plt.xlabel('Actual Reimbursement Amount')
        plt.ylabel('Predicted Reimbursement Amount')
        plt.legend()
        plt.grid(True)
        plt.show()

# Main execution logic
if __name__ == "__main__":
    model_instance = ReimbursementModel()
    model_instance.load_or_train_model()

    if len(sys.argv) == 2 and sys.argv[1] == "--train":
        model_instance.train_and_evaluate_model()
    elif len(sys.argv) == 4:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        reimbursement = model_instance.calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        print(reimbursement)
    else:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        print("       python3 calculate_reimbursement.py --train")
