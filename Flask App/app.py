from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load models
with open('models/resnet_buy_model_2.pkl', 'rb') as f:
    resnet_buy_model = pickle.load(f)
with open('models/resnet_sell_model_2.pkl', 'rb') as f:
    resnet_sell_model = pickle.load(f)

# Load and preprocess dataset
sp = pd.read_csv('sp.csv')
scaler = RobustScaler()
X = sp.drop(columns=['ideal_buy_date_expanded', 'ideal_sell_date_expanded'])
scaler.fit(X)  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input date from the request
    data = request.get_json()
    input_date = data.get('date')  # Expecting a date in the request payload

    # Ensure the date exists in the dataset
    if input_date not in sp.index:
        return jsonify({'error': 'Date not found in the dataset'}), 400

    # Extract features for the given date
    features = sp.loc[input_date].values.reshape(1, -1)  # Reshape for a single prediction

    # Scale features
    features_scaled = scaler.transform(features)

    # Make predictions using ResNet models
    buy_proba = resnet_buy_model.predict_proba(features_scaled)[:, 1][0]
    sell_proba = resnet_sell_model.predict_proba(features_scaled)[:, 1][0]

    return jsonify({
        'buy_probability': buy_proba,
        'sell_probability': sell_proba
    })

@app.route('/run_analysis', methods=['GET'])
def run_analysis():

    def ideal_date_single(binary_col, position: int):
        result = pd.Series(0, index=binary_col.index)
        blockof1s = []

        for idx, val in binary_col.items():
            if val == 1:
                blockof1s.append(idx)
            elif blockof1s:
                pos = max(min(position, len(blockof1s) - 1), 0)
                selected_idx = blockof1s[pos]
                result.at[selected_idx] = 1
                blockof1s = []

        if blockof1s:
            pos = max(min(position, len(blockof1s) - 1), 0)
            selected_idx = blockof1s[pos]
            result.at[selected_idx] = 1

        return result

    sp['ideal_buy_date_single'] = ideal_date_single(sp['ideal_buy_date_expanded'], 2)
    sp['ideal_sell_date_single'] = ideal_date_single(sp['ideal_sell_date_expanded'], 2)

    buy_dates = sp[sp['ideal_buy_date_single'] == 1].index
    sell_dates = sp[sp['ideal_sell_date_single'] == 1].index

    # Ensure same length
    n_trades = min(len(buy_dates), len(sell_dates))
    cumulative_return = 1.0

    for i in range(n_trades):
        buy_price = sp.loc[buy_dates[i], 'close']
        sell_price = sp.loc[sell_dates[i], 'close']
        trade_return = (sell_price - buy_price) / buy_price
        cumulative_return *= (1 + trade_return)

    # Cleanup
    drop_cols = ['ideal_buy_date_single', 'ideal_sell_date_single']
    sp.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Calculate cumulative return
    cumulative_return = np.round((cumulative_return - 1) * 100, 2)

    return jsonify({
        'cumulative_return': f"{cumulative_return}%"
    })

if __name__ == '__main__':
    app.run(debug=True)


