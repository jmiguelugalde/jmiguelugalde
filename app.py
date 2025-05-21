from flask import Flask, request, jsonify, send_from_directory
from services.file_parser import parse_file
from services.forecasting_service import AVAILABLE_MODELS
import pandas as pd
import os
import datetime # Keep for potential future use, though not directly in this combined version

# --- App Setup ---
# static_folder='static' ensures Flask knows where to find static files for send_from_directory
# It's good practice to define it here if you have a dedicated static folder at the root.
# If 'static' is a blueprint's static folder, it's handled differently.
# For this simple app, 'static' at the root is fine.
app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store parsed data (list of dictionaries)
parsed_data_store = None 

# --- Static File Serving ---
# Serves index.html from the 'static' folder for the root URL
@app.route('/')
def serve_index():
    # app.static_folder is automatically set to 'static' from Flask app initialization
    return send_from_directory(app.static_folder, 'index.html')

# Serves any other static files (like CSS, JS referenced by index.html if they were separate)
# While index.html has inline JS/CSS, this route is good practice.
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# --- API Endpoints ---
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads for sales data.
    Uses services.file_parser.parse_file to process the file.
    Stores parsed data in global `parsed_data_store`.
    """
    global parsed_data_store
    df_temp = None # Temporary variable to hold DataFrame to assist in file cleanup logic
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400

    # It's good practice to secure the filename, e.g., using werkzeug.utils.secure_filename
    # For this example, we'll use the original filename directly but acknowledge this.
    filename = file.filename 
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)
        df_temp = parse_file(file_path) # df_temp holds the parsed DataFrame or None
        
        if df_temp is not None:
            # Convert DataFrame to list of dictionaries for storage
            parsed_data_store = df_temp.to_dict(orient='records') 
            return jsonify({
                "message": f"File '{filename}' processed successfully. {len(df_temp)} rows processed.",
                "fileName": filename,
                "rowCount": len(df_temp)
            }), 200
        else:
            # parse_file returned None, indicating a parsing error handled within parse_file
            if os.path.exists(file_path): # Attempt to remove the problematic file
                os.remove(file_path)
            # The error message from parse_file is printed on the server, 
            # here we give a generic message to the client.
            return jsonify({"error": f"Failed to parse file '{filename}'. Check file format (CSV, Excel, TXT) and content (first column dates, second sales). See server logs for details."}), 400
    except Exception as e:
        app.logger.error(f"Error processing file {filename}: {e}", exc_info=True)
        if os.path.exists(file_path): # Clean up file if an unexpected error occurs
            os.remove(file_path)
        return jsonify({"error": f"An unexpected error occurred while processing file '{filename}': {str(e)}"}), 500
    finally:
        # Ensure cleanup of the saved file if parsing was successful and df_temp is not None
        if df_temp is not None and os.path.exists(file_path):
             os.remove(file_path)
        # If df_temp is None (parsing failed), file might have been removed already in the 'else' block.
        # If an exception occurred before df_temp was assigned, it's handled in 'except'.

@app.route('/models', methods=['GET'])
def get_available_models():
    """
    Returns a list of available forecasting model names.
    Retrieves model names from services.forecasting_service.AVAILABLE_MODELS.
    """
    try:
        model_names = list(AVAILABLE_MODELS.keys())
        return jsonify({"models": model_names}), 200
    except Exception as e:
        app.logger.error(f"Error retrieving model list: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve model list."}), 500

@app.route('/forecast', methods=['POST'])
def forecast_data():
    """
    Generates sales forecast based on previously uploaded data.
    Expects JSON body with 'model_name' and 'horizon'.
    Uses services.forecasting_service for the actual forecasting.
    Returns JSON with historical and forecasted data.
    """
    global parsed_data_store
    if parsed_data_store is None:
        return jsonify({"error": "No data available for forecasting. Please upload a file first via /upload endpoint."}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
        
    model_name = data.get('model_name')
    horizon_str = data.get('horizon')

    if not model_name or horizon_str is None:
        return jsonify({"error": "Missing 'model_name' or 'horizon' in request body."}), 400

    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": f"Model '{model_name}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}"}), 400

    try:
        horizon = int(horizon_str)
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
    except ValueError:
        return jsonify({"error": "Invalid horizon. Horizon must be a positive integer."}), 400

    try:
        # Reconstruct DataFrame from stored data (list of dicts)
        df = pd.DataFrame(parsed_data_store)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        df.set_index('fecha', inplace=True)
        
        df['ventas'] = pd.to_numeric(df['ventas'], errors='coerce')
        df.dropna(subset=['ventas'], inplace=True)

        if df.empty or 'ventas' not in df.columns:
             return jsonify({"error": "No valid sales data to forecast after processing. Ensure data has 'fecha' and 'ventas' columns with valid entries."}), 400

        sales_series = df['ventas'].sort_index() # Sort by date before forecasting
        
        # Ensure the series has a frequency; this is critical for time series models.
        # The forecasting services also attempt this, but good to be robust.
        if sales_series.index.freq is None:
            inferred_freq = pd.infer_freq(sales_series.index)
            if inferred_freq:
                sales_series = sales_series.asfreq(inferred_freq)
            else:
                app.logger.warning("Could not infer frequency for sales data in app.py; defaulting to 'D'. This may affect forecast accuracy if data is not daily.")
                sales_series = sales_series.asfreq('D') # Default if inference fails
            
            # Fill NaNs that might have been introduced by asfreq
            sales_series.fillna(method='ffill', inplace=True) 
            sales_series.fillna(method='bfill', inplace=True) 
            if sales_series.isnull().all(): # Check if series became all NaNs
                return jsonify({"error": "Sales data is all NaN after frequency adjustment and filling. Cannot proceed."}), 500

        forecasting_function = AVAILABLE_MODELS[model_name]
        # Forecasting functions return (historical_series, forecast_series) or (series, None) on error
        historical_series, forecast_results = forecasting_function(sales_series, horizon)

        if forecast_results is not None: # Check if forecast was successful
            # Prepare historical data for JSON response (using the potentially modified historical_series)
            historical_data_json = [{"date": idx.strftime('%Y-%m-%d %H:%M:%S'), "sales": val} 
                                    for idx, val in historical_series.items()]
            
            # Prepare forecasted data for JSON response
            forecasted_data_json = [{"date": idx.strftime('%Y-%m-%d %H:%M:%S'), "forecasted_value": val} 
                                     for idx, val in forecast_results.items()]
            
            return jsonify({
                "model_used": model_name,
                "horizon_periods": horizon,
                "historical_data": historical_data_json,
                "forecasted_data": forecasted_data_json
            }), 200
        else:
            # This implies an error within the forecasting function (e.g., series too short, model fit error)
            # Specific error details are logged by the service functions.
            error_msg = f"Failed to generate forecast using {model_name}. " \
                        "The series might be too short, data incompatible with the model, or other model fitting issues occurred. " \
                        "Check server logs for more details from the forecasting service."
            app.logger.error(f"Forecasting function {model_name} returned None for forecast part. Series length: {len(sales_series)}")
            return jsonify({"error": error_msg}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during forecasting in app.py: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during forecasting: {str(e)}"}), 500

if __name__ == '__main__':
    # Make host and port configurable, e.g., via environment variables for deployment flexibility
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(debug=True, host=host, port=port)
