# Flight_Delay_Prediction
🚀 A Streamlit web app that predicts flight delays using a machine learning model. Input flight details, view delay predictions, and understand total delay causes. Built with Python, Pandas, Scikit-learn, and Streamlit.
# ✈️ Flight Delay Predictor

A machine learning-powered Streamlit web app that predicts whether a flight will be delayed based on input features like weather, airline, aircraft issues, and more.

## 🚀 Features

- Predict flight delays using a trained machine learning model.
- Input features such as:
  - Year, Month
  - Carrier, Airport (encoded)
  - Delay counts and minutes from various causes (weather, NAS, etc.)
  - Cancelled/diverted flights
- Displays:
  - Raw model output
  - Total delay minutes
  - Final classification: Delayed / On Time / Not Delayed

## 🧠 ML Model

- Trained on historical flight data
- Preprocessed using Scikit-Learn
- Pickled model and scaler are loaded dynamically

## 🛠️ Tech Stack

- **Python**
- **Pandas**, **Scikit-learn**
- **Streamlit** for the UI

## 📁 Project Structure
