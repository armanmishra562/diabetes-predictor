# 🩺 Diabetes Risk Predictor

This project is a simple Streamlit web application that predicts the risk of diabetes using machine learning models trained on the PIMA Indian Diabetes Dataset.

## 📦 Features

- Predict diabetes risk based on medical parameters
- Trained with multiple ML models, best model selected (Random Forest)
- Clean and simple web UI using Streamlit
- Ready to deploy on Render or any cloud platform

## 📊 Dataset

- [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features include Glucose, Blood Pressure, BMI, Age, etc.

## 📁 Project Structure

```
diabetes-predictor/
├── app.py
├── diabetes_prediction.py
├── best_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
```

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
python diabetes_prediction.py  # To train the model and save it
streamlit run app.py           # To start the Streamlit app
```

## 🌐 Deployment on Render

- Push the project to GitHub
- Create a new Web Service on Render
- Use the following settings:

```
Build Command: pip install -r requirements.txt
Start Command: streamlit run app.py --server.port=$PORT --server.enableCORS=false
```

## 🛠 Dependencies

- streamlit
- scikit-learn
- pandas
- numpy
- joblib

## ✨ Author

Arman Kumar Mishra
