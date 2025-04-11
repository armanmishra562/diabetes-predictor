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
├── model.py
├── best_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
```

## 🚀 How to Run Locally

```bash
git clone https://github.com/armanmishra562/diabetes-predictor.git
cd diabetes-predictor
pip install -r requirements.txt
python model.py  # To train the model and save it
streamlit run app.py           # To start the Streamlit app
```

## 🌐 Deployment on Render

👉 [Click here to open the app](https://diabetes-predictor-29ho.onrender.com)

## 🛠 Dependencies

- streamlit
- scikit-learn
- pandas
- numpy
- joblib

## ✨ Author

Arman Kumar Mishra
