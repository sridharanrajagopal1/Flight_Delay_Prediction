import streamlit as st
import pandas as pd
import pickle
import os


# Load Model and Scaler
def load_model_and_scaler():
    try:
        base_path = os.path.dirname(__file__)  # Get current file directory
        model_path = os.path.join(base_path, '..', 'models', 'flight_delay_model.pkl')
        scaler_path = os.path.join(base_path, '..', 'models', 'scaler.pkl')

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None


# Prediction Function
def predict_delay(input_data_df, model, scaler):
    try:
        scaled_data = scaler.transform(input_data_df)
        prediction = model.predict(scaled_data)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    st.set_page_config(page_title="Flight Delay Predictor", layout="wide")
    st.title('✈️ Flight Delay Predictor')
    
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return

    st.sidebar.header('📝 Enter Flight Info')

    # Collect User Input for Required Features
    input_data = {
        'year': st.sidebar.number_input('Year', min_value=2000, max_value=2030, value=2023),
        'month': st.sidebar.number_input('Month', min_value=1, max_value=12, value=8),
        'carrier': st.sidebar.number_input('Carrier (encoded)', min_value=0),
        'carrier_name': st.sidebar.number_input('Carrier Name (encoded)', min_value=0),
        'airport': st.sidebar.number_input('Airport (encoded)', min_value=0),
        'airport_name': st.sidebar.number_input('Airport Name (encoded)', min_value=0),
        'arr_flights': st.sidebar.number_input('Arrival Flights', min_value=0),
        'carrier_ct': st.sidebar.number_input('Carrier Delay Count', min_value=0),
        'weather_ct': st.sidebar.number_input('Weather Delay Count', min_value=0),
        'nas_ct': st.sidebar.number_input('NAS Delay Count', min_value=0),
        'security_ct': st.sidebar.number_input('Security Delay Count', min_value=0),
        'late_aircraft_ct': st.sidebar.number_input('Late Aircraft Delay Count', min_value=0),
        'arr_cancelled': st.sidebar.number_input('Arrival Cancelled', min_value=0),
        'arr_diverted': st.sidebar.number_input('Arrival Diverted', min_value=0),
        'arr_delay': st.sidebar.number_input('Arrival Delay (minutes)', min_value=0),
        'carrier_delay': st.sidebar.number_input('Carrier Delay (minutes)', min_value=0),
        'weather_delay': st.sidebar.number_input('Weather Delay (minutes)', min_value=0),
        'nas_delay': st.sidebar.number_input('NAS Delay (minutes)', min_value=0),
        'security_delay': st.sidebar.number_input('Security Delay (minutes)', min_value=0),
        'late_aircraft_delay': st.sidebar.number_input('Late Aircraft Delay (minutes)', min_value=0),
        'delayed': st.sidebar.number_input('Delayed Count', min_value=0),
        'delay_percentage': st.sidebar.number_input('Delay Percentage', min_value=0.0, max_value=100.0, step=0.1),
        'total_delay_causes': st.sidebar.number_input('Total Delay Causes', min_value=0)
    }

    input_data_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        prediction = predict_delay(input_data_df, model, scaler)

        # Total delay in minutes
        total_delay_minutes = (
            input_data['carrier_delay'] +
            input_data['weather_delay'] +
            input_data['nas_delay'] +
            input_data['security_delay'] +
            input_data['late_aircraft_delay']
        )

        # Final decision rule
        if total_delay_minutes < 15:
            status = "🟢 Not Delayed"
        elif total_delay_minutes == 15:
            status = "🕒 On Time"
        else:
            status = "✈️ Delayed"

        # Display results
        if prediction is not None:
            st.subheader("🧠 Raw Model Output")
            st.write(f"Prediction: {'Delayed' if prediction[0] == 1 else 'Not Delayed'}")

            st.subheader("📊 Total Delay Minutes")
            st.write(f"Total Delay: `{total_delay_minutes}` minutes")

            st.subheader("✅ Final Classification (Based on Rule)")
            st.success(f"Result: {status}")
        else:
            st.error("Prediction could not be completed.")

if __name__ == '__main__':
    main()
