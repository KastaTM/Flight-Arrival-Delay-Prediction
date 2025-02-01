# Flight-Arrival-Delay-Prediction

📌 Project: Flight Delay Prediction

This project aims to implement and apply Big Data concepts using Apache Spark to predict the arrival delay of commercial flights based on features known at takeoff.


# Big Data: Spark Practical Work

## Polytechnic University of Madrid
**School of Computer Engineering**

### Master in Data Science - Group 23
**Rodrigo Castañón Martínez, Ignacio Domínguez Calvo, Dakota James Mellish**

## 📌 Project: Flight Delay Prediction
This project aims to implement and apply Big Data concepts using Apache Spark to predict the arrival delay of commercial flights based on features known at takeoff.

---

## 📁 Repository Structure
```
📂 BigData_Spark_Project
├── 📁 data                  # Flight and aircraft datasets
├── 📁 notebooks             # Notebooks for exploratory analysis and modeling
├── 📁 models                # Saved trained models
├── 📁 scripts               # Processing and Spark application scripts
├── 📁 results               # Model evaluation results and metrics
├── app.py                   # Spark application for predictions
├── models.py                # Model definition and training
└── README.md                # Documentation file
```

---

## 🔧 Technologies Used
- **Apache Spark** (Spark Core, Spark SQL, MLlib)
- **Python** (Pandas, NumPy, Matplotlib, Scikit-learn)
- **Jupyter Notebook** for exploratory analysis
- **Linux** for execution and development

---

## 📊 Data Used
The dataset consists of publicly available domestic flight data from the **US Department of Transportation**, accessible through Dataverse.

The selected features include flight details, airline information, delays, airport codes, and aircraft characteristics. Various transformations and preprocessing steps were applied to enhance data quality.

---

## 🚀 Project Implementation
### 1️⃣ Data Exploration and Feature Selection
- Dataset analysis and selection of relevant variables
- Data transformation and preprocessing
- Feature engineering (creation of derived variables like "IsWeekend," "FlightDuration," etc.)

### 2️⃣ Model Training
- Models used:
  - **Linear Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Gradient Boosting Trees**
- Evaluation using **cross-validation** (5-fold)
- Hyperparameter tuning with Grid Search

### 3️⃣ Model Validation
- **Metrics used:** RMSE, R², MAE, MAPE
- The **Gradient Boosting Tree Regression** model achieved the best performance with the lowest RMSE.

### 4️⃣ Apache Spark Application
- Developed **app.py** script to execute predictions on new data.
- Loads pre-trained model and applies it to new records.
- Generates reports with predictions and evaluation metrics.

---

## 📜 Usage Instructions
### 🔹 Install Dependencies
Run in a virtual environment:
```bash
pip install -r requirements.txt
```
### 🔹 Execute the Spark Application
Ensure Apache Spark is installed and configured.
```bash
spark-submit app.py data/2007.csv
```
**Expected Output:**
- Files in `results/` containing predictions and evaluation metrics.

---

## 📌 Conclusions
- Successfully developed a scalable Apache Spark application for flight delay prediction.
- **Gradient Boosting Tree Regression** delivered the best predictive performance.
- The application enables predictions on new data and model evaluation.
- Future improvements may include incorporating weather data and expanding the dataset across multiple years.

---

## 📧 Contact
For any inquiries regarding this project, contact:
**Rodrigo Castañón Martínez, Ignacio Domínguez Calvo, Dakota James Mellish**

We hope this documentation is helpful! 🚀

