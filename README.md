# ✈️ SkySense: AI-Powered Flight Delay Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

> **Note:** Above badges represent the tech stack used in this project.

## 📖 Overview
**SkySense** is an end-to-end Machine Learning web application designed to predict flight delays in real-time. By leveraging historical aviation data and weather patterns, the system utilizes **Ensemble Learning techniques (Random Forest)** to provide travelers with accurate delay probabilities.

This project moves beyond simple analysis, offering a fully interactive **Streamlit Dashboard** where users can input flight parameters and get instant risk assessments.

## 🚀 Key Features
- **Real-time Prediction Engine:** Instant delay probability calculation based on user inputs.
- **Interactive Dashboard:** Built with **Streamlit** for a seamless user experience.
- **Data Analytics:** Visual insights into airline performance and seasonal delay trends.
- **Scalable Architecture:** Modular code structure ready for cloud deployment (Render/AWS).

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python-based UI)
- **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Render / Streamlit Cloud

## 📂 Project Structure

├── app.py                # Main Streamlit Web Application
├── model_build.py        # Script to train and serialize the ML model
├── xgbmodel.pkl          # Pre-trained Model file (Generated)
├── requirements.txt      # Project dependencies
├── README.md             # Project Documentation
└── .gitignore            # Files to ignore (Datasets, Secrets)


## Usage
1. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

2. Initialize Model Pipeline
```
python model_build.py
```


3. Launch Application
```
streamlit run app.py
```

4. Evaluate the trained model's performance.

5. Predict flight delays for new data using the trained model by loading the pickle file.

## Conclusion

### 📊 Performance Metrics
The model was evaluated on a held-out validation set (20% split).

--Accuracy: 83.5%**
--Precision: Optimized to reduce false positives.**
--Robustness: The Random Forest ensemble reduces the risk of overfitting compared to single Decision Trees.

👨‍💻 Author
**[GUDDY THAKUR] Full Stack Developer & ML Enthusiast**
