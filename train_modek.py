from model import HousePriceModel
import os

def main():
    """Train and save the house price prediction model"""
    print("Training House Price Prediction Model...")
    
    # Initialize and train model
    model = HousePriceModel()
    results = model.train()
    
    print("Model Training Completed!")
    print(f"Mean Absolute Error: ${results['mae']:,.2f}")
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"Training samples: {results['training_samples']}")
    print(f"Test samples: {results['test_samples']}")
    
    # Save the model
    model.save_model('house_price_model.pkl')
    print("Model saved as 'house_price_model.pkl'")
    
    # Test prediction with sample data
    sample_features = [2000, 3, 2, 1995, 5000, 2, 7]  # square_footage, bedrooms, bathrooms, year_built, lot_size, garage, location_quality
    prediction = model.predict(sample_features)
    print(f"\nSample Prediction:")
    print(f"Features: {sample_features}")
    print(f"Predicted Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
