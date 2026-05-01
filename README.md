# Heart Disease Prediction

This project is a machine learning based Heart Disease Prediction application. It uses a trained Random Forest pipeline saved as `pipe_rf.pkl` and provides a simple Streamlit web interface where a user can enter patient health details and get a heart disease risk prediction.

## Author

**Name:** Harmanav Singh  
**Roll Number:** 2210990369

## Project Files

- `app_h.py` - Streamlit web application.
- `pipe_rf.pkl` - Trained machine learning model pipeline.
- `heart.csv` - Heart disease dataset used for analysis/training.
- `heart_disease_prediction.ipynb` - Jupyter Notebook containing analysis and model work.
- `requirements.txt` - Python dependencies required to run the app.
- `correlationfigure.png` - Correlation heatmap image.
- `featureplot.png` - Feature distribution plot image.

## Requirements

Install Python 3.10 before running this project.

Required Python packages:

```txt
streamlit
pandas
scikit-learn==1.4.2
```

Important: keep `scikit-learn==1.4.2` because the saved model file `pipe_rf.pkl` was created using this version.

## How To Run

Open PowerShell or the VS Code terminal and go to the project folder:

```powershell
cd "D:\HARMANAV\Heart_Disease_Prediction"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run app_h.py
```

After running the command, open this URL in your browser:

```text
http://localhost:8501
```

## How To Run In VS Code

1. Open VS Code.
2. Click **File > Open Folder**.
3. Select the `Heart_Disease_Prediction` folder.
4. Open the terminal using **Terminal > New Terminal**.
5. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_h.py
```

6. Open `http://localhost:8501` in a browser.

## Application Usage

1. Enter patient health information such as age, sex, chest pain type, cholesterol level, blood pressure, ECG result, and other values.
2. Click **Predict Heart Disease Risk**.
3. The app displays the probability of heart disease and no heart disease.
4. The final result is shown as either high risk or low risk.

## Notes

- This project is for academic and learning purposes.
- The prediction is based on a machine learning model and should not be treated as medical advice.
- For any real health concern, consult a qualified medical professional.

## Troubleshooting

If `streamlit` is not recognized, make sure the virtual environment is activated:

```powershell
.\.venv\Scripts\activate
```

If package installation fails, upgrade pip:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `streamlit==1.57.0` is not available on another system, use:

```txt
streamlit==1.50.0
pandas
scikit-learn==1.4.2
```

## Copyright

Copyright (c) 2026 Harmanav Singh, Roll Number 2210990369.

All rights reserved. This project and its contents are submitted for academic purposes. Unauthorized copying, redistribution, or commercial use without permission is not allowed.
