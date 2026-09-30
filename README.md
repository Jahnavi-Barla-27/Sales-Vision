# Sales Prediction Using Machine Learning

## 1. Project Overview

This project is a Machine Learning based Sales Prediction system. The main objective of this project is to predict sales using historical sales data.

The project uses machine learning algorithms to analyze sales-related features such as quantity, discount, shipping cost, and year.

Two machine learning models are used in this project:

- Linear Regression
- Random Forest Regressor

The models are evaluated and compared using MAE, MSE, and R2 Score to identify the better-performing model.

---

## 2. Objectives

The main objectives of this project are:

- To analyze historical sales data.
- To preprocess and prepare the dataset.
- To identify important features related to sales.
- To train machine learning models for sales prediction.
- To compare Linear Regression and Random Forest models.
- To predict sales for new input data.
- To visualize sales trends and model performance.

---

## 3. Dataset

The project uses the `SuperStoreOrders.csv` dataset.

The dataset contains information related to orders, customers, products, regions, sales, quantity, discount, shipping cost, and year.

Important features used for prediction are:

- Quantity
- Discount
- Shipping Cost
- Year

Target variable:

- Sales

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

---

## 5. Machine Learning Models

### Linear Regression

Linear Regression is used to predict sales based on the relationship between the input features and the target variable.

### Random Forest Regressor

Random Forest Regressor uses multiple decision trees to make predictions and can capture more complex relationships in the data.

---

## 6. Project Workflow

The project follows these steps:

1. Load the sales dataset.
2. Display the first five rows.
3. Check column names.
4. Check missing values.
5. Convert the Sales column into numeric format.
6. Select input features.
7. Split the data into training and testing sets.
8. Train the Linear Regression model.
9. Train the Random Forest Regressor model.
10. Generate predictions.
11. Evaluate both models.
12. Compare model performance.
13. Select the best-performing model.
14. Predict sales for new data.
15. Visualize sales and prediction results.

---

## 7. Model Evaluation

The models are evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted sales.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted sales.

### R2 Score

Measures how well the model explains the variation in the sales data.

A higher R2 Score indicates better model performance.

---

## 8. Data Visualizations

The project includes visualizations such as:

- Actual Sales vs Linear Regression Predictions
- Total Sales by Category
- Total Sales by Region
- Top 10 Products by Sales
- Total Sales by Year
- Model Comparison using R2 Score
- Feature Importance in Sales Prediction

These visualizations help understand sales patterns and model performance.

---

## 9. Sales Prediction

The trained model can predict sales for new input values based on:

- Quantity
- Discount
- Shipping Cost
- Year

Example input:

- Quantity: 5
- Discount: 0.10
- Shipping Cost: 20
- Year: 2026

The model generates the predicted sales value for these inputs.

---

## 10. Project Structure

```text
Sales_Prediction_Project
│
├── SuperStoreOrders.csv
├── sales_prediction.py
├── requirements.txt
└── README.md