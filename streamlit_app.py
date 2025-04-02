import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# Set page title and layout
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("Gold Price Prediction App")

# Define paths for model and scaler
MODEL_PATH = 'gold_price_model.h5'
SCALER_PATH = 'gold_price_scaler.pkl'

# Functions for data preparation and model training
def load_and_prepare_data():
    df = pd.read_csv('gold_price_data.csv')
    
    # Format and clean the data
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort dates in ascending order
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Check if Price column needs cleaning
    if df['Price'].dtype == object:
        df['Price'] = df['Price'].replace({',': ''}, regex=True)
        df['Price'] = df['Price'].astype('float64')
    
    # Set Date as index for easier time series operations
    df_price = df.set_index('Date')
    
    return df_price

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(df_price):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_price['Price'].values.reshape(-1, 1))
    
    # Split data into train and test sets
    seq_length = 7
    split_percent = 0.8
    split_point = int(len(scaled_data) * split_percent)
    
    train_data = scaled_data[:split_point]
    test_data = scaled_data[split_point:]
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Create and train model
    input_shape = (seq_length, X_train.shape[2])
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Add a status update while training
    progress_text = st.empty()
    progress_text.text("Training model... This may take a few minutes.")
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    progress_text.text("Model training complete!")
    
    # Save the model and scaler
    model.save(MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, history

# Check if model and scaler exist and load or train accordingly
@st.cache_resource
def get_model_and_scaler(df_price):
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # Load existing model and scaler
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler, None
        except Exception as e:
            st.warning(f"Error loading existing model: {e}. Training a new model.")
    
    # Train a new model if files don't exist or error loading
    return train_model(df_price)

# Function to prepare data for prediction
def prepare_prediction_data(date, data, seq_length, scaler):
    # Find the most recent data before the prediction date
    input_data = data[data.index < pd.to_datetime(date)].tail(seq_length)
    
    # If we don't have enough historical data
    if len(input_data) < seq_length:
        st.warning(f"Not enough historical data. Need {seq_length} data points.")
        return None
    
    # Prepare the input sequence
    features = input_data['Price'].values.reshape(-1, 1)
    
    # Scale the features using the same scaler used during training
    scaled_features = scaler.transform(features)
    
    # Reshape for LSTM: [samples, time steps, features]
    x_input = np.array([scaled_features])
    
    return x_input

# Function to make prediction
def predict_gold_price(model, input_data, scaler):
    if input_data is None:
        return None
    
    prediction = model.predict(input_data)
    
    # Reshape prediction for inverse transform
    # Create a dummy array with zeros for other columns (if scaler was fit on multiple columns)
    dummy = np.zeros((len(prediction), 1))
    prediction_reshaped = np.concatenate((prediction, dummy), axis=1)
    
    # Inverse transform to get the actual price value
    prediction_unscaled = scaler.inverse_transform(prediction_reshaped)[:,0]
    
    return prediction_unscaled[0]

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")

# Date input
prediction_date = st.sidebar.date_input(
    "Select a date to predict gold price:",
    datetime.date.today()
)

# Main app logic
try:
    # Load and prepare data
    df_price = load_and_prepare_data()
    
    # Get or train model
    model, scaler, history = get_model_and_scaler(df_price)
    
    # Display training information if new model was trained
    if history is not None:
        st.subheader("Model Training Results")
        fig_loss = plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        st.pyplot(fig_loss)
    
    # Display historical data
    st.subheader("Historical Gold Price Data")
    st.line_chart(df_price[['Price']])
    
    # Make prediction when button is clicked
    if st.sidebar.button("Predict Gold Price"):
        # Define your sequence length (same as used in training)
        seq_length = 7
        
        # Prepare data for prediction
        input_data = prepare_prediction_data(prediction_date, df_price, seq_length, scaler)
        
        if input_data is not None:
            # Make prediction with progress indicator
            with st.spinner('Making prediction...'):
                predicted_price = predict_gold_price(model, input_data, scaler)
            
            if predicted_price is not None:
                # Display result
                st.success(f"Predicted Gold Price for {prediction_date}: ${predicted_price:.2f}")
                
                # Create a simple visualization
                st.subheader("Prediction Visualization")
                recent_data = df_price.tail(30).copy()
                
                # Add the prediction point
                prediction_df = pd.DataFrame(
                    {'Price': [predicted_price]}, 
                    index=[pd.to_datetime(prediction_date)]
                )
                
                # Create combined plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(recent_data.index, recent_data['Price'], label='Historical')
                ax.plot(prediction_df.index, prediction_df['Price'], 'ro', label='Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Gold Price (USD)')
                ax.legend()
                st.pyplot(fig)
                
                # Add some context about the prediction
                last_price = df_price['Price'].iloc[-1]
                change = predicted_price - last_price
                percent_change = (change / last_price) * 100
                
                st.info(f"""
                **Analysis:**
                - Last known price: ${last_price:.2f}
                - Predicted change: ${change:.2f} ({percent_change:.2f}%)
                """)
    
    # Extra: let user see recent historical data
    st.sidebar.subheader("View Historical Data")
    days_to_show = st.sidebar.slider("Number of days to display:", 10, 100, 30)
    
    st.subheader(f"Last {days_to_show} days of historical data")
    st.dataframe(df_price.tail(days_to_show))

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check your data file and make sure it contains the required columns.")

# Add some additional information
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
This application predicts the price of gold based on historical data using a deep learning model.
""")