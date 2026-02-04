from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open('LinearRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. {e}")
    model = None

# Global variables for dropdowns
companies = []
car_models = []
years = []
fuel_types = []

def load_config():
    global companies, car_models, years, fuel_types
    if not model:
        return

    try:
        # Based on inspection:
        # Pipeline Steps: ['columntransformer', 'linearregression']
        # Transformer: onehotencoder, Columns: ['name', 'company', 'fuel_type']
        # Category 0: name (models)
        # Category 1: company
        # Category 2: fuel_type
        
        if hasattr(model, 'steps'):
            # Find the step with transformers (usually the first one in this case)
             for name, step in model.steps:
                if hasattr(step, 'transformers_'):
                    for t_name, t_transform, t_cols in step.transformers_:
                        if t_name == 'onehotencoder' and hasattr(t_transform, 'categories_'):
                            categories = t_transform.categories_
                            # Safer mapping based on length if strict index fails, but strict index is 0, 1, 2
                            if len(categories) >= 3:
                                car_models = list(categories[0])
                                companies = list(categories[1])
                                fuel_types = list(categories[2])
                                
        # Set years (standard range)
        years = sorted(list(range(2000, 2025)), reverse=True)
        
        # Sort lists for better UI
        companies.sort()
        car_models.sort()
        fuel_types.sort()
        
        print(f"Loaded {len(companies)} companies, {len(car_models)} models, {len(fuel_types)} fuel types.")
        
    except Exception as e:
        print(f"Error extracting categories: {e}")
        # Fallback
        companies = ['Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Hindustan', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Land', 'Mahindra', 'Maruti', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
        fuel_types = ['Diesel', 'LPG', 'Petrol']

load_config()

@app.route('/')
def index():
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded", 500
    
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = request.form.get('kilo_driven')
        
        # Clean kms_driven (remove commas, ensuring it's an int)
        if hasattr(kms_driven, 'replace'):
             kms_driven = int(kms_driven.replace(',', ''))
        else:
             kms_driven = int(kms_driven)

        # Create DataFrame with columns in the exact order the model expects
        # Feature Names In: ['name', 'company', 'year', 'kms_driven', 'fuel_type']
        data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], 
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        prediction = model.predict(data)
        
        # Handle potential valid output format
        if len(prediction) > 0:
            price = np.round(prediction[0], 2)
            # Ensure no negative prices
            if price < 0:
                price = 0  # Or handle as error
            return str(price)
        else:
             return "Error in prediction"
             
    except Exception as e:
        print(f"Prediction Error: {e}")
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
