# Diabetes Prediction Web Application

A machine learning project that predicts the likelihood of a person having diabetes using the **Support Vector Machine (SVM)** classifier. The trained model is saved and then deployed as a user-friendly web application using the **Streamlit** framework.

This repository serves as a practical implementation of an end-to-end machine learning workflow, from data preprocessing and model training to deployment.

-----

## ✨ Key Features

  * **SVM Classification:** Utilizes the Support Vector Machine (SVM) algorithm with a **linear kernel** for robust binary classification (Diabetic / Non-Diabetic).
  * **Data Preprocessing:** Implements crucial steps including **Train-Test Split** to prepare the data for optimal model performance.
  * **Model Persistence:** The trained classifier is saved using the **`pickle`** library, allowing the web application to load and use the model without retraining.
  * **Interactive Web App:** Deploys the model using **Streamlit**, providing a clean, interactive interface where users can input 8 medical parameters and receive an instant diagnosis.

-----

## 📊 Dataset

The project uses the publicly available **Pima Indians Diabetes Dataset**.

  * **File Name:** `diabetesDataset.csv`
  * **Size:** 768 patient records (rows)
  * **Target Variable:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)
  * **Input Features (8 total):**

| Feature | Description |
| :--- | :--- |
| `Pregnancies` | Number of times pregnant. |
| `Glucose` | Plasma glucose concentration (2 hours in an oral glucose tolerance test). |
| `BloodPressure` | Diastolic blood pressure (mm Hg). |
| `SkinThickness` | Triceps skin fold thickness (mm). |
| `Insulin` | 2-Hour serum insulin (mu U/ml). |
| `BMI` | Body mass index (weight in $kg/height$ in $m^2$). |
| `DiabetesPedigreeFunction` | A function that scores the likelihood of diabetes based on family history. |
| `Age` | Age (years). |

-----

## 🛠️ Technical Stack and Dependencies

The project is built using the following core technologies:

  * **Python**
  * **Development Environment:** **Jupyter Notebook** (Used for the ML workflow).
  * **ML Libraries:**
      * `scikit-learn` (Model training and preprocessing)
      * `pandas` & `numpy` (Data handling and array manipulation)
      * `pickle` (Model saving/loading)
  * **Deployment:** `streamlit` (Web application framework)

-----

## 📁 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `DiabeticPrediction.ipynb` | The Jupyter Notebook containing the end-to-end ML development code (EDA, SVM Training, Evaluation). |
| `DiabeticPredictionWebApp.py` | The Python script that loads the trained model and runs the Streamlit web application. |
| `diabetic_trained_model.sav` | The serialized **Support Vector Machine (SVM)** model, saved using the `pickle` library. |
| `diabetesDataset.csv` | The Pima Indians Diabetes Dataset used for training. |
| `requirements.txt` | Lists all necessary Python dependencies for easy setup. |

-----

## ⚙️ Installation and Setup

### 1\. Clone the Repository

```bash
git clone https://github.com/Rishita560/Diabetes-Prediction
cd Diabetes-Prediction
```

### 2\. Install Dependencies

It is highly recommended to use a virtual environment.

**Create `requirements.txt`:**

```
numpy
pandas
scikit-learn
streamlit
jupyter
```

**Install the libraries:**

```bash
# Optional: Create a virtual environment
# python -m venv venv
# source venv/bin/activate 

# Install required packages
pip install -r requirements.txt
```

-----

## 🚀 How to Run the Project

You have two options for interacting with this project: using the live, **deployed web application** or running the **code locally**.

-----

### A. 🌐 Option 1: Access the Live Web Application (Recommended)

The easiest way to use the predictive model is via the application deployed on the Streamlit Community Cloud.

| Item | Link |
| :--- | :--- |
| **Live App URL** | **[CLICK HERE TO USE THE DIABETES PREDICTION APP](https://diabetes-prediction-ml-model-web.streamlit.app/)** |

#### How to Use the App:

1.  Click on the link above.
2.  The application will load in your browser.
3.  Enter the **8 medical parameters** in the input fields provided.
4.  Click the **`Diabetes Test Result`** button to get the model's prediction (Outcome: **1** - Diabetic / **0** - Non-Diabetic).

-----

### B. 💻 Option 2: Run the Project Locally

To reproduce the model training or run the web application on your own machine, follow these steps:

#### 1\. Run the Model Training Workflow (Jupyter Notebook)

This step trains the Support Vector Machine (SVM) model and saves it for the web app to use.

1.  Start the Jupyter Notebook server in your terminal (after installing the dependencies from `requirements.txt`):
    ```bash
    jupyter notebook
    ```
2.  Open **`DiabeticPrediction.ipynb`** in your browser.
3.  **Run all cells** in the notebook. The script will perform the full ML workflow and save the final trained model as **`diabetic_trained_model.sav`** in the root directory.

#### 2\. Launch the Web Application (Streamlit)

Once you have the **`diabetic_trained_model.sav`** file, you can launch the interactive web application:

1.  Ensure the necessary files are in your project root directory: **`DiabeticPredictionWebApp.py`**, **`diabetic_trained_model.sav`**, and **`diabetesDataset.csv`**.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run DiabeticPredictionWebApp.py
    ```
3.  The application will automatically open in your default web browser.
