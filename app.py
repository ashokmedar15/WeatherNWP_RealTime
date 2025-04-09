from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.losses import MeanSquaredError
import matplotlib
import matplotlib.pyplot as plt
import os
import socket

# Configure matplotlib to avoid GUI conflicts
matplotlib.use('Agg')

app = Flask(__name__)

# Define paths
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'convlstm_weather_model.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
IMAGE_PATH = os.path.join(STATIC_FOLDER, 'prediction_heatmap.png')

# Load model and scaler
try:
    model = load_model(MODEL_PATH, custom_objects={'MeanSquaredError': MeanSquaredError})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")


# Placeholder for predict_heatmap function (replace with real-time data logic post-MVP)
def predict_heatmap(date_str, time_str):
    """
    Generate a heatmap based on date and time using a dummy input for MVP.
    Replace with actual data fetching and preprocessing post-MVP.
    Returns (success, description) tuple.
    """
    try:
        # Parse date and time
        input_date = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')

        # Remove the 3-day restriction for MVP
        # if input_date < datetime.now() - timedelta(days=3):
        #     return False, "Real-time data available only for the last 3 days."

        # Dummy input for MVP (shape: (batch, time, latitude, longitude, variables) = (1, 24, 17, 25, 2))
        # Replace with actual data loading (e.g., from .npy or ERA5) post-MVP
        X_input = np.zeros((1, 24, 17, 25, 2))

        # Generate prediction
        y_pred = model.predict(X_input)
        y_pred_reshaped = y_pred.squeeze(-1).reshape(-1, 1)
        dummy = np.zeros((y_pred_reshaped.shape[0], 1))
        y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred_reshaped, dummy]))[:, 0]
        prediction = y_pred_unscaled.reshape(17, 25)

        # Create and save heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(prediction, cmap='coolwarm')
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Predicted Temperature for {date_str} {time_str}')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')

        # Ensure static folder exists
        os.makedirs(STATIC_FOLDER, exist_ok=True)
        plt.savefig(IMAGE_PATH, dpi=100, bbox_inches='tight')
        plt.close()

        # Dynamic description with explicit line breaks
        description = (
            f"Prediction Details for {date_str} {time_str}:\n"
            f"- This heatmap represents the predicted temperature distribution over Bengaluru,\n"
            f"  generated using the ConvLSTM model based on historical weather patterns.\n"
            f"- The color scale indicates temperature in °C, with warmer colors representing\n"
            f"  higher temperatures.\n"
            f"- Note: This is a dummy prediction for MVP; real-time data will be integrated\n"
        f"  post-MVP."
        )
        return True, description
    except Exception as e:
        print(f"Prediction error: {e}")
        return False, f"Error generating prediction: {e}"


# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    image_path = None
    description = None

    if request.method == 'POST':
        date_str = request.form['date']
        time_str = request.form['time']

        # Server-side validation
        try:
            input_date = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
        except ValueError:
            error = "Invalid date/time format. Use YYYY-MM-DD HH:MM."
        else:
            success, result = predict_heatmap(date_str, time_str)
            if success:
                image_path = 'prediction_heatmap.png'
                description = result
            else:
                error = result

    return render_template('index.html', error=error, image_path=image_path, description=description)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


# Error handling
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('index.html', error="Internal server error. Please try again later."), 500


if __name__ == '__main__':
    from waitress import serve
    import os

    print("Starting waitress server...")
    port = int(os.getenv('PORT', '5000'))  # Use PORT env variable or default to 5000
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server IP: {local_ip}. Access via http://{local_ip}:{port}/ or http://127.0.0.1:{port}/")

    try:
        # Check if port is in use and fallback if needed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', port))
        if result == 0:
            print(f"Port {port} in use, trying 5001...")
            port = 5001
        sock.close()
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error starting server: {e}")