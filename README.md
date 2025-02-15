# 🩺 Heart Disease Analysis
## **Predicting Heart Disease Using Machine Learning**  

This project aims to develop a **machine learning model** to predict the likelihood of heart disease based on patient health data. The analysis involves **data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation** to determine the most effective classification algorithm.

---

## 📊 Dataset
- The dataset contains multiple **health-related attributes** used to predict heart disease.
- Features include **age, cholesterol levels, blood pressure, heart rate, and more**.

---

## ⚡ Project Features
✔ **Data Preprocessing**: Handled missing values, feature scaling, and encoding categorical data.  
✔ **Exploratory Data Analysis (EDA)**: Identified key risk factors and correlations using **visualizations**.  
✔ **Machine Learning Models**: Implemented and compared six classification models:
  - **Logistic Regression** *(Best Accuracy: 91.21%)*
  - **Naïve Bayes** *(Best AUROC Score: 0.948)*
  - **Support Vector Machine (SVM)**
  - **Random Forest**
  - **Decision Tree**
  - **K-Nearest Neighbors (KNN)*
✔ **Model Evaluation**: Compared performance using **accuracy, precision, recall, F1-score, and AUROC metrics**.  
✔ **Visualization**: Used **Matplotlib** and **Seaborn** to generate insights into heart disease trends.  

---

## 🔬 Results
### 🏆 Model Performance Based on Accuracy
| Model                   | Accuracy |
|-------------------------|----------|
| **Logistic Regression** | **91.21%** |
| Naïve Bayes            | 87.91% |
| SVM                    | 85.71% |
| Random Forest          | 84.62% |
| Decision Tree          | 75.82% |
| K-Nearest Neighbors (KNN) | 60.44% |

### 🎯 Model Performance Based on AUROC Score
| Model                   | AUROC Score |
|-------------------------|-------------|
| **Naïve Bayes**        | **0.948** |
| Random Forest          | 0.939 |
| Logistic Regression    | 0.938 |
| SVM                    | 0.935 |
| Decision Tree          | 0.760 |
| K-Nearest Neighbors (KNN) | 0.665 |

💡 **Key Takeaways:**  
- **Logistic Regression** achieved the **highest accuracy (91.21%)**.  
- **Naïve Bayes** performed best based on **AUROC score (0.948)**.  
- **Random Forest and SVM** showed strong predictive power.  

---

## 💻 Technologies Used
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  
- **Development Environment**: Jupyter Notebook  
