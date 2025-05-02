from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


app = Flask(__name__)

# Load models
with open('models/rf_buy_model.pkl', 'rb') as f:
    rf_buy_model = pickle.load(f)
with open('models/rf_sell_model.pkl', 'rb') as f:
    rf_sell_model = pickle.load(f)

# Load and preprocess data
sp_trim_rf = pd.read_csv('sp_trim_rf.csv')
sp_trim_rf['date'] = pd.to_datetime(sp_trim_rf['date'])
sp_trim_rf.set_index('date', inplace=True)

medians = sp_trim_rf.median().to_dict()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', medians=medians)

def ideal_buy_sell(sp_trim_rf):
    """
    Calculate the ideal buy/sell strategy return and extract relevant dates and prices.
    """
    sp_trim_rf_copy = sp_trim_rf.copy()
    first_close_value = 57.33000183105469
    sp_trim_rf_copy['close'] = sp_trim_rf_copy['close'].cumsum() + first_close_value

    buy_dates = sp_trim_rf_copy[sp_trim_rf_copy['buy_signal'] == 1].index.tolist()
    sell_dates = sp_trim_rf_copy[sp_trim_rf_copy['sell_signal'] == 1].index.tolist()

    cumulative_return = 1.0
    for buy, sell in zip(buy_dates, sell_dates):
        buy_price = sp_trim_rf_copy.loc[buy, 'close']
        sell_price = sp_trim_rf_copy.loc[sell, 'close']
        trade_return = (sell_price - buy_price) / buy_price
        cumulative_return *= (1 + trade_return)

    ideal_return = (cumulative_return - 1) * 100
    close_prices = sp_trim_rf_copy['close'].tolist() if 'close' in sp_trim_rf_copy.columns else []
    return ideal_return, buy_dates, sell_dates, close_prices

def ideal_date_single(binary_col, position: int):
    """
    For each block of consecutive 1s in binary_col, select one index (based on position)
    and return a Series with 1 at only those selected indices.
    """
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

@app.route('/predict_user_input', methods=['POST'])
def predict_user_input():
    """
    Generate predictions and compute ideal buy/sell strategy return, holding strategy return, 
    and model-based strategy return for visualization.
    """
    model_features = rf_buy_model.feature_names_in_
    X_df = sp_trim_rf[model_features].iloc[-1:].copy()

    buy_proba = rf_buy_model.predict_proba(X_df)[0][1] * 100
    sell_proba = rf_sell_model.predict_proba(X_df)[0][1] * 100

    # Generate buy_signal and sell_signal using ideal_date_single
    sp_trim_rf['buy_signal'] = ideal_date_single(sp_trim_rf['ideal_buy_date_expanded'], 2)
    sp_trim_rf['sell_signal'] = ideal_date_single(sp_trim_rf['ideal_sell_date_expanded'], 2)

    # Generate ideal_buy_date_single and ideal_sell_date_single for plotting
    sp_trim_rf['ideal_buy_date_single'] = ideal_date_single(sp_trim_rf['ideal_buy_date_expanded'], 2)
    sp_trim_rf['ideal_sell_date_single'] = ideal_date_single(sp_trim_rf['ideal_sell_date_expanded'], 2)


    # Ideal Buy/Sell Strategy
    ideal_return, buy_dates, sell_dates, close_prices = ideal_buy_sell(sp_trim_rf.copy())

    # Holding Strategy
    sp_trim_rf_copy = sp_trim_rf.copy()
    first_close_value = 57.33000183105469
    sp_trim_rf_copy['close'] = sp_trim_rf_copy['close'].cumsum() + first_close_value

    start_date = sp_trim_rf_copy.index[0]
    end_date = sp_trim_rf_copy.index[-1]
    start_price = 57.33000183105469
    end_price = sp_trim_rf_copy.loc[end_date, 'close']
    holding_return = ((end_price - start_price) / start_price) * 100
    holding_return = np.round(holding_return, 2)

    # Prepare data for holding chart
    holding_chart = {
        'x': sp_trim_rf_copy.index.tolist(),
        'y': sp_trim_rf_copy['close'].tolist()
    }

    # Model-Based Strategy
    model_buy_dates = sp_trim_rf_copy[sp_trim_rf['buy_signal'] == 1].index.tolist()
    model_sell_dates = sp_trim_rf_copy[sp_trim_rf['sell_signal'] == 1].index.tolist()

    # Ensure same length for buy/sell dates
    n_trades = min(len(model_buy_dates), len(model_sell_dates))
    model_cumulative_return = 1.0

    for i in range(n_trades):
        buy_price = sp_trim_rf_copy.loc[model_buy_dates[i], 'close']
        sell_price = sp_trim_rf_copy.loc[model_sell_dates[i], 'close']
        if buy_price == 0:  # Avoid division by zero
            continue
        trade_return = (sell_price - buy_price) / buy_price
        model_cumulative_return *= (1 + trade_return)

    model_return = (model_cumulative_return - 1) * 100
    model_return = np.round(model_return, 2)

    plt.figure(figsize=(15, 8))
    plt.plot(sp_trim_rf_copy.index, sp_trim_rf_copy['close'], label='Close Price')
    plt.plot(sp_trim_rf_copy[sp_trim_rf_copy['ideal_buy_date_single'] == 1].index, sp_trim_rf_copy[sp_trim_rf_copy['ideal_buy_date_single'] == 1]['close'], '^', color='green', label='Ideal Buy Date')
    plt.plot(sp_trim_rf_copy[sp_trim_rf_copy['ideal_sell_date_single'] == 1].index, sp_trim_rf_copy[sp_trim_rf_copy['ideal_sell_date_single'] == 1]['close'], 'v', color='red', label='Ideal Sell Date')
    plt.title('Ideal Buy/Sell Dates')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    img_ideal = io.BytesIO()
    plt.savefig(img_ideal, format='png')
    img_ideal.seek(0)
    ideal_chart_base64 = base64.b64encode(img_ideal.getvalue()).decode('utf-8')
    plt.close()

    # Generate the holding strategy chart
    plt.figure(figsize=(15, 8))
    plt.plot(sp_trim_rf_copy.index, sp_trim_rf_copy['close'], label='Holding Strategy')
    plt.title('Holding Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    img_holding = io.BytesIO()
    plt.savefig(img_holding, format='png')
    img_holding.seek(0)
    holding_chart_base64 = base64.b64encode(img_holding.getvalue()).decode('utf-8')
    plt.close()

    # Generate the model-based strategy chart
    plt.figure(figsize=(15, 8))
    plt.plot(sp_trim_rf_copy.index, sp_trim_rf_copy['close'], label='Close Price')
    plt.scatter(model_buy_dates, [sp_trim_rf_copy.loc[date, 'close'] for date in model_buy_dates], color='blue', label='Model Buy', marker='^')
    plt.scatter(model_sell_dates, [sp_trim_rf_copy.loc[date, 'close'] for date in model_sell_dates], color='orange', label='Model Sell', marker='v')
    plt.title('Model-Based Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    img_model = io.BytesIO()
    plt.savefig(img_model, format='png')
    img_model.seek(0)
    model_chart_base64 = base64.b64encode(img_model.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({
        'holding_return': holding_return,
        'ideal_return': round(ideal_return, 2),
        'model_return': model_return,
        'ideal_chart_image': ideal_chart_base64,
        'holding_chart_image': holding_chart_base64,
        'model_chart_image': model_chart_base64
    })

@app.route('/macro_input_form', methods=['POST'])
def macro_input_form():
    """
    Predict buy and sell probabilities based on user-defined macroeconomic variables.
    Automatically fill missing features with medians.
    """
    user_input_features = [
        'HOUST', 'DFF', 'SAHMCURRENT', 'volume',
        'rsi_56', 'bb_upper_200', 'bb_lower_200',
        'bb_upper_50', 'bb_lower_50'
    ]

    input_data = medians.copy()
    for feature in user_input_features:
        if feature in request.form:
            input_data[feature] = float(request.form.get(feature, medians.get(feature, 0)))

    X_df = pd.DataFrame([input_data])
    X_df = X_df[rf_buy_model.feature_names_in_]

    buy_proba = rf_buy_model.predict_proba(X_df)[0][1] * 100
    sell_proba = rf_sell_model.predict_proba(X_df)[0][1] * 100

    return jsonify({
        'buy_probability': round(buy_proba, 4),
        'sell_probability': round(sell_proba, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)