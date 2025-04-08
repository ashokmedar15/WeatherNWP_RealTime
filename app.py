from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.losses import MeanSquaredError
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# import cdsapi  # Commented out for MVP
# from netCDF4 import Dataset  # Commented out for MVP

app = Flask(__name__)

# Load model and scaler
try:
    model = load_model('convlstm_weather_model.h5', custom_objects={'MeanSquaredError': MeanSquaredError})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")


@app.route('/', methods=['GET', 'POST'])
def predict():
    image_path = None
    if request.method == 'POST':
        date_str = request.form['date']
        time_str = request.form['time']
        try:
            input_date = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
        except ValueError:
            return "Invalid date/time format. Use YYYY-MM-DD HH:MM."

        if input_date < datetime.now() - timedelta(days=3):
            return "Real-time data available only for the last 3 days."

        # Temporary dummy input for MVP (replace with real-time data post-MVP)
        # Shape: (batch, time, latitude, longitude, variables) = (1, 24, 17, 25, 2)
        X_input = np.zeros((1, 24, 17, 25, 2))

        # Commented out CDS API and preprocessing for MVP
        '''
        # Fetch real-time ERA5 data (to be implemented post-MVP)
        c = cdsapi.Client()
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['2m_temperature', 'surface_pressure'],
                    'year': str(datetime.now().year),
                    'month': str(datetime.now().month).zfill(2),
                    'day': str(datetime.now().day).zfill(2),
                    'time': [f'{h:02d}:00' for h in range(24)],
                    'area': [13.17, 77.50, 12.97, 77.75],  # Bengaluru bounds (lat max, lon min, lat min, lon max)
                    'format': 'netcdf',
                },
                'era5_bengaluru_latest.nc'
            )
            print("ERA5 data fetched successfully.")
        except Exception as e:
            return f"Error fetching ERA5 data: {e}"

        # Preprocess data (to be implemented post-MVP)
        try:
            dataset = Dataset('era5_bengaluru_latest.nc', 'r')
            temp = dataset.variables['t2m'][:] - 273.15  # Shape: (24, 17, 25)
            pressure = dataset.variables['sp'][:] / 100  # Shape: (24, 17, 25)
            X_input = np.stack((temp, pressure), axis=-1)  # Initial shape mismatch
            # Adjust shape to (1, 24, 17, 25, 2) - to be finalized post-MVP
            X_input = np.transpose(X_input, (1, 0, 2, 3, 4))  # Placeholder adjustment
            X_input = scaler.transform(X_input.reshape(-1, 2)).reshape(1, 24, 17, 25, 2)
        except Exception as e:
            return f"Error preprocessing data: {e}"
        '''

        # Generate prediction with dummy input
        try:
            y_pred = model.predict(X_input)
            y_pred_reshaped = y_pred.squeeze(-1).reshape(-1, 1)
            dummy = np.zeros((y_pred_reshaped.shape[0], 1))
            y_pred_unscaled = scaler.inverse_transform(np.hstack([y_pred_reshaped, dummy]))[:, 0]
            prediction = y_pred_unscaled.reshape(17, 25)

            plt.figure(figsize=(10, 6))
            plt.imshow(prediction, cmap='coolwarm')
            plt.colorbar(label='Temperature (°C)')
            plt.title(f'Predicted Temperature for {date_str} {time_str}')
            plt.xlabel('Longitude Index')
            plt.ylabel('Latitude Index')
            image_path = os.path.join(app.root_path, 'static', 'prediction_heatmap.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            plt.savefig(image_path)
            plt.close()
        except Exception as e:
            return f"Error generating prediction: {e}"

    return render_template('index.html', image_path=image_path)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    from waitress import serve

    print("Starting waitress server on 0.0.0.0:5000...")
    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server IP: {local_ip}. Access via http://{local_ip}:5000/ or http://127.0.0.1:5000/")
    try:
        # Check if port 5000 is in use and fallback to 5001 if needed
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', 5000))
        if result == 0:
            print("Port 5000 in use, trying 5001...")
            port = 5001
        else:
            port = 5000
        sock.close()
        serve(app, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error starting server: {e}")