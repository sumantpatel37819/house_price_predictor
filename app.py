from flask import Flask, request, jsonify
from flask_cors import CORS
from model import HousePriceModel
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
house_model = HousePriceModel()

def load_model():
    """Load the trained model if it exists"""
    try:
        if os.path.exists('house_price_model.pkl'):
            house_model.load_model('house_price_model.pkl')
            print("Model loaded successfully")
        else:
            print("No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        'message': 'House Price Predictor API',
        'endpoints': {
            '/predict': 'POST - Predict house price',
            '/health': 'GET - API health check',
            '/model-info': 'GET - Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': house_model.is_trained
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Random Forest Regressor',
        'features': [
            'square_footage',
            'bedrooms', 
            'bathrooms',
            'year_built',
            'lot_size',
            'garage',
            'location_quality (1-10 scale)'
        ],
        'model_trained': house_model.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'square_footage', 'bedrooms', 'bathrooms', 'year_built',
            'lot_size', 'garage', 'location_quality'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract features
        features = [
            data['square_footage'],
            data['bedrooms'],
            data['bathrooms'], 
            data['year_built'],
            data['lot_size'],
            data['garage'],
            data['location_quality']
        ]
        
        # Validate feature types and ranges
        try:
            features = [float(f) for f in features]
        except ValueError:
            return jsonify({
                'error': 'All features must be numeric values'
            }), 400
        
        # Basic validation for reasonable values
        if features[0] < 500 or features[0] > 10000:  # square_footage
            return jsonify({
                'error': 'Square footage should be between 500 and 10000'
            }), 400
            
        if features[6] < 1 or features[6] > 10:  # location_quality
            return jsonify({
                'error': 'Location quality should be between 1 and 10'
            }), 400
        
        # Make prediction
        if not house_model.is_trained:
            return jsonify({
                'error': 'Model not trained. Please train the model first.'
            }), 503
        
        predicted_price = house_model.predict(features)
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'features': {
                'square_footage': features[0],
                'bedrooms': features[1],
                'bathrooms': features[2],
                'year_built': features[3],
                'lot_size': features[4],
                'garage': features[5],
                'location_quality': features[6]
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to train the model (optional)"""
    try:
        results = house_model.train()
        
        # Save the trained model
        house_model.save_model('house_price_model.pkl')
        
        return jsonify({
            'message': 'Model trained successfully',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Training failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    print("Starting House Price Predictor API...")
    print("Visit http://localhost:5000 for API information")
    app.run(debug=True, host='0.0.0.0', port=5000)
