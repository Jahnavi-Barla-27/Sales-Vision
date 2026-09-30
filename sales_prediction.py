import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the dataset
df = pd.read_csv("SuperStoreOrders.csv", encoding="latin1")

# 2. Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())


# 3. Display column names
print("\nColumn Names:")
print(df.columns.tolist())


# 4. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 5. Convert Sales column to numeric
# This removes commas from values such as "1,648"
df["sales"] = (
    df["sales"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)


# 6. Select input features
# These columns are available in your dataset
features = [
    "quantity",
    "discount",
    "shipping_cost",
    "year"
]

X = df[features]
y = df["sales"]


# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. Create Linear Regression model
linear_model = LinearRegression()

# 9. Train Linear Regression model
linear_model.fit(X_train, y_train)

# 10. Make Linear Regression predictions
linear_pred = linear_model.predict(X_test)


# 11. Create Random Forest model
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 12. Train Random Forest model
random_forest_model.fit(X_train, y_train)

# 13. Make Random Forest predictions
random_forest_pred = random_forest_model.predict(X_test)


# 11. Display actual and predicted sales
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Linear Regression Prediction": linear_pred,
    "Random Forest Prediction": random_forest_pred
})

print("\nActual Sales vs Predicted Sales:")
print(results.head(10))


# 14. Evaluate Linear Regression
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)


# 15. Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, random_forest_pred)
rf_mse = mean_squared_error(y_test, random_forest_pred)
rf_r2 = r2_score(y_test, random_forest_pred)


# 16. Display Model Comparison
print("\n========== MODEL COMPARISON ==========")

print("\nLinear Regression:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("R2 Score:", linear_r2)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)


# 17. Find the best model
if rf_r2 > linear_r2:
    print("\nBest Model: Random Forest Regressor")
else:
    print("\nBest Model: Linear Regression")

# 13. Predict sales for new data
new_data = pd.DataFrame({
    "quantity": [5],
    "discount": [0.10],
    "shipping_cost": [20],
    "year": [2026]
})

predicted_sales = linear_model.predict(new_data)

print("\nPredicted Sales for New Data:")
print("Quantity:", new_data["quantity"].iloc[0])
print("Discount:", new_data["discount"].iloc[0])
print("Shipping Cost:", new_data["shipping_cost"].iloc[0])
print("Year:", new_data["year"].iloc[0])
print("Predicted Sales:", predicted_sales[0])


# 14. Create a graph
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    linear_pred
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Linear Regression Predictions")

plt.show()
# ==========================================
# DATA VISUALIZATION
# ==========================================

# 1. Total Sales by Category
category_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Total Sales by Region
region_sales = df.groupby("region")["sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
region_sales.plot(kind="bar")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. Top 10 Products by Sales
top_products = (
    df.groupby("product_name")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_products.plot(kind="bar")
plt.title("Top 10 Products by Sales")
plt.xlabel("Product Name")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 4. Sales by Year
year_sales = df.groupby("year")["sales"].sum()

plt.figure(figsize=(8, 5))
year_sales.plot(kind="bar")
plt.title("Total Sales by Year")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Sales by Year
year_sales = df.groupby("year")["sales"].sum()

plt.figure(figsize=(8, 5))
year_sales.plot(kind="bar")
plt.title("Total Sales by Year")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==========================================
# MODEL COMPARISON VISUALIZATION
# ==========================================

# 5. Compare Model R2 Scores

model_names = [
    "Linear Regression",
    "Random Forest"
]

r2_scores = [
    linear_r2,
    rf_r2
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)

plt.title("Model Comparison - R2 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.show()


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

# 6. Feature Importance using Random Forest

feature_importance = pd.Series(
    random_forest_model.feature_importances_,
    index=features
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_importance.plot(kind="bar")

plt.title("Feature Importance in Sales Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()