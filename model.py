import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic house price data for demonstration"""
        np.random.seed(42)
        
        data = {
            'square_footage': np.random.randint(800, 4000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples),
            'lot_size': np.random.randint(1000, 10000, n_samples),
            'garage': np.random.randint(0, 3, n_samples),
            'location_quality': np.random.randint(1, 10, n_samples)  # 1-10 scale
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic prices based on features
        base_price = (
            df['square_footage'] * 150 +
            df['bedrooms'] * 10000 +
            df['bathrooms'] * 15000 +
            (df['year_built'] - 1950) * 500 +
            df['lot_size'] * 5 +
            df['garage'] * 8000 +
            df['location_quality'] * 20000
        )
        
        # Add some noise
        noise = np.random.normal(0, 50000, n_samples)
        df['price'] = base_price + noise
        
        return df
    
    def train(self, df=None):
        """Train the model on the provided data"""
        if df is None:
            df = self.generate_sample_data()
        
        # Prepare features and target
        features = ['square_footage', 'bedrooms', 'bathrooms', 'year_built', 
                   'lot_size', 'garage', 'location_quality']
        X = df[features]
        y = df['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'mae': mae,
            'r2_score': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict(self, features):
        """Predict house price for given features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to numpy array and scale
        features_array = np.array([features]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        return float(prediction)
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
