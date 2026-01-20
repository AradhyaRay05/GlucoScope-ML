# 🩺 GlucoScope-ML - AI-powered Diabetes Classification

---

## 🔍 Project Goal
GlucoScope-ML is designed to predict the **likelihood of diabetes** using **machine learning models**. It provides a robust framework for analyzing patient data and predicting diabetes outcomes based on key health metrics. The goal of this project is to leverage **data-driven insights** to assist in early diagnosis and better management of diabetes.

---

## 📖 Overview
The project utilizes a **real-world diabetes dataset** to train and evaluate machine learning models. It includes data preprocessing, exploratory data analysis (EDA), model building, and evaluation. By integrating preprocessing, model training, and evaluation, GlucoScope-ML demonstrates how AI can contribute to healthcare advancements.

---

## 🔄 **Project Workflow**

### **1️⃣ Data Preprocessing & EDA**
- **Data Inspection:** Loaded the dataset with Pandas for an initial inspection of its structure and types.  
- **EDA & Visualization:** Visualized feature distributions and correlations using Seaborn and Matplotlib.  
- **Missing Value Imputation:** Replaced missing values in key columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with mean or median values.  
- **Feature Scaling:** Standardized the data range using `StandardScaler`.  
- **Train-Test Split:** Divided the processed data into training and testing sets for model validation.  

---

### **2️⃣ Model Building**
Tested and compared multiple models to find the best performer:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machines (SVM)  

---

### **3️⃣ Evaluation Metrics**
- **Accuracy:** The Random Forest model achieved an accuracy of **98.75%** on the test set and **99.94%** on the training set.
- **Confusion Matrix:** Visualized to analyze the distribution of true positives, true negatives, false positives, and false negatives.  
- **Classification Report:** Provided precision, recall, and F1-score for each class to evaluate model performance comprehensively.  
- **Cross-Validation:** Achieved an average accuracy of **95%** across 5 folds, ensuring model robustness.  

---

## 🌐 **Deployment**

The project has been deployed using **Streamlit**, providing an interactive web-based interface for real-time diabetes predictions. Users can input health metrics such as glucose levels, blood pressure, and BMI to receive instant predictions. 

---

## 🛠 **Tech Stack**

- **Python** – The core programming language used for data analysis, model building, and deployment.  
- **Pandas / NumPy** – For efficient data manipulation, cleaning, and numerical computations.  
- **Scikit-learn** – For preprocessing, implementing machine learning models, and evaluating their performance.  
- **Seaborn / Matplotlib** – For creating insightful visualizations to understand data distributions and relationships.  
- **Joblib** – For saving and loading the trained models and preprocessing pipelines.  
- **Streamlit** – For building and deploying an interactive web application for real-time predictions.

---

## 📂 Project Structure
```
StartuPredict/
├── diabetes.csv                            # Raw dataset used for training
├── .gitignore                              # Files/directories to exclude from Git 
├── LICENSE                                 # Allows reuse, with attribution,no warranty
├── README.md                               # Project documentation
├── app.py                                  # Main Streamlit app
├── model.pkl                               # Trained MLR model
├── scaler.pkl                              # Pre-fitted StandardScaler object for input normalization
├── requirements.txt                        # Required dependencies
└── Diabetes_Classification.ipynb           # Notebook for training and testing
```
---

## ✨ **Features**

- Predicts diabetes likelihood based on health metrics.  
- Supports multiple machine learning models.  
- Provides insights into feature importance and model performance.  

---

## 🚀 **Future Enhancements**

- Integration with a web-based interface for real-time predictions.  
- Addition of advanced models like XGBoost and CatBoost.  
- Deployment on cloud platforms for scalability.  

---

## 🧪 **How to Run Locally**

```
# Clone the repository
git clone https://github.com/AradhyaRay05/GlucoScope-ML.git

# Navigate to the project directory
cd GlucoScope-ML

# Install the dependencies
pip install -r requirements.txt
```
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

<p>
  <a href="mailto:aradhyaray99@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
  <a href="www.linkedin.com/in/rayaradhya"><img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
  <a href="https://github.com/AradhyaRay05"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
</p>

---

Thanks for visiting ! Feel free to explore my other repositories and connect with me. 🚀