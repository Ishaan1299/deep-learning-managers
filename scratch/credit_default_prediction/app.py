import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# 1. Define the exact same architecture to load the saved weights
class CreditRiskANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CreditRiskANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# 2. Load the Model Globally
try:
    print("Loading model for API...")
    input_dimension = 93 # As trained
    num_classes = 4     # P1, P2, P3, P4
    
    model = CreditRiskANN(input_dimension, num_classes)
    model.load_state_dict(torch.load('saved_models/ann_model.pth', weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    
    # We also need a sample to get the correct column names for the dummy request
    sample_df = pd.read_csv('data/processed/X_test.csv', nrows=1)
    feature_columns = sample_df.columns.tolist()

except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        
        # In a real app, you would take raw HTML inputs, apply the StandardScaler, 
        # and One-Hot Encode them exactly as in data_prep.py.
        # For this interactive demo, we will take a random row from our preprocessed X_test set,
        # perturb it slightly based on the UI sliders to show dynamic changes.
        
        # 1. Grab a random row from the Test set to act as our "Base Customer"
        base_customer = pd.read_csv('data/processed/X_test.csv').sample(1)
        
        # 2. Apply the UI Slider adjustments (pretending these are standardized scale adjustments)
        # We find the closest matching column names for the UI inputs
        if 'income_modifier' in data:
            base_customer.iloc[0, 0] += float(data['income_modifier']) # Just tweaking the first feature as a proxy
            
        if 'cibil_modifier' in data:
            base_customer.iloc[0, 1] -= float(data['cibil_modifier']) # Tweak second feature as proxy

        # 3. Predict PyTorch
        input_tensor = torch.tensor(base_customer.values, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0].numpy()
            predicted_class = torch.argmax(output, dim=1).item()
            
        # 4. Format Response
        risk_labels = ['P1 (Excellent)', 'P2 (Good)', 'P3 (Moderate Risk)', 'P4 (High Risk / Default)']
        
        response = {
            'prediction': risk_labels[predicted_class],
            'prediction_index': predicted_class,
            'probabilities': {
                'P1': float(probabilities[0] * 100),
                'P2': float(probabilities[1] * 100),
                'P3': float(probabilities[2] * 100),
                'P4': float(probabilities[3] * 100)
            },
            'status': 'success'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
