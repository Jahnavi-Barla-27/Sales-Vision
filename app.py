import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Sales Vision — Sales Prediction & Analytics System 📊")

st.caption("🤖 Machine Learning • 📈 Sales Analytics • 🔮 Smart Predictions")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("SuperStoreOrders.csv", encoding="latin1")

# Convert sales to numeric
df["sales"] = (
    df["sales"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)


# ==========================================
# TRAINING DATA
# ==========================================

features = [
    "quantity",
    "discount",
    "shipping_cost",
    "year"
]

X = df[features]
y = df["sales"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)


# ==========================================
# RANDOM FOREST
# ==========================================
random_forest_model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=1
)

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)

rf_r2 = r2_score(y_test, random_forest_pred)
rf_mae = mean_absolute_error(y_test, random_forest_pred)
rf_mse = mean_squared_error(y_test, random_forest_pred)


# ==========================================
# SELECT BEST MODEL
# ==========================================

if rf_r2 > linear_r2:
    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_r2 = rf_r2
else:
    best_model = linear_model
    best_model_name = "Linear Regression"
    best_r2 = linear_r2


# ==========================================
# DASHBOARD METRICS
# ==========================================

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Model", best_model_name)
col2.metric("R² Score", f"{best_r2:.3f}")
col3.metric("Total Sales", f"{df['sales'].sum():,.2f}")
col4.metric("Total Records", len(df))

st.write("### 📉 Error Metrics")

error_col1, error_col2 = st.columns(2)

error_col1.metric(
    "MAE",
    f"₹{(rf_mae if best_model_name == 'Random Forest' else linear_mae):,.2f}"
)

error_col2.metric(
    "RMSE",
    f"₹{((rf_mse if best_model_name == 'Random Forest' else linear_mse) ** 0.5):,.2f}"
)
# ==========================================
# SALES PREDICTION
# ==========================================

st.subheader("🔮 Predict Future Sales")

col1, col2 = st.columns(2)
# ==============================
# SESSION STATE
# ==============================

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None

if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
with col1:
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=5
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

with col2:
    shipping_cost = st.number_input(
        "Shipping Cost",
        min_value=0.0,
        value=20.0
    )

    year = st.number_input(
        "Year",
        min_value=2020,
        max_value=2035,
        value=2026
    )

if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None

if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = None
# ==========================================
# PREDICTION BUTTON
# ==========================================

# ==========================================================
# INITIALIZE SESSION STATE
# ==========================================================

if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None

if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = None


# ==========================================================
# PREDICT SALES
# ==========================================================

if st.button("🔮 Predict Sales", key="predict_sales_button"):

    # Create input data
    new_data = pd.DataFrame({
        "quantity": [quantity],
        "discount": [discount],
        "shipping_cost": [shipping_cost],
        "year": [year]
    })

    # Make prediction
    predicted_sales = float(
        best_model.predict(new_data)[0]
    )

    
    # Save previous and current prediction

    previous_prediction = st.session_state.current_prediction

    st.session_state.previous_prediction = previous_prediction
    st.session_state.current_prediction = predicted_sales

    # ==========================================
# AUTOMATIC OBSERVATION
# ==========================================

st.subheader("💡 Automatic Observation")

previous = st.session_state.get("previous_prediction")
current = st.session_state.get("current_prediction")

if previous is not None and current is not None:

    difference = current - previous

    if previous != 0:
        percentage_change = (difference / previous) * 100
    else:
        percentage_change = 0

    if difference > 0:
        st.success(
            f"📈 Predicted sales increased by "
            f"₹{difference:,.2f} "
            f"({percentage_change:+.2f}%) compared with the previous scenario."
        )

    elif difference < 0:
        st.warning(
            f"📉 Predicted sales decreased by "
            f"₹{abs(difference):,.2f} "
            f"({percentage_change:+.2f}%) compared with the previous scenario."
        )

    else:
        st.info(
            "➡️ Predicted sales remained unchanged compared with the previous scenario."
        )

else:
    st.info(
        "Make two predictions with different inputs to see the automatic observation."
    )
        
  # ==========================================
# WHAT CHANGED?
# ==========================================

st.subheader("🔄 What Changed?")

previous = st.session_state.get("previous_prediction")
current = st.session_state.get("current_prediction")

if previous is not None and current is not None:

    difference = current - previous

    if difference > 0:
        st.success(
            f"📈 Sales increased by ₹{difference:,.2f} "
            f"compared with the previous prediction."
        )

    elif difference < 0:
        st.warning(
            f"📉 Sales decreased by ₹{abs(difference):,.2f} "
            f"compared with the previous prediction."
        )

   
# ==========================================
# PREDICTION VISUALIZATION
# ==========================================

st.subheader("📊 Prediction Visualization")

previous = st.session_state.get("previous_prediction")
current = st.session_state.get("current_prediction")

if previous is not None and current is not None:

    prediction_data = pd.DataFrame({
        "Prediction": ["Previous", "Current"],
        "Sales": [previous, current]
    })

    # Set color according to prediction change
    if current > previous:
        bar_colors = ["gray", "green"]

    elif current < previous:
        bar_colors = ["gray", "red"]

    else:
        bar_colors = ["gray", "blue"]

    fig = px.bar(
        prediction_data,
        x="Prediction",
        y="Sales",
        title="Previous vs Current Sales Prediction"
    )

    fig.update_traces(
        marker_color=bar_colors
    )

    fig.update_layout(
        xaxis_title="Prediction",
        yaxis_title="Predicted Sales (₹)",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    difference = current - previous

    if difference > 0:
        st.success(
            f"📈 Sales increased by ₹{difference:,.2f}"
        )

    elif difference < 0:
        st.warning(
            f"📉 Sales decreased by ₹{abs(difference):,.2f}"
        )

    else:
        st.info(
            "➡️ Sales prediction remained unchanged."
        )

else:

    st.info(
        "🔄 Make two predictions with different inputs "
        "to see the comparison chart."
    )


 # ==========================================
# SALES ANALYTICS
# ==========================================

st.subheader("📊 Sales Analytics")

# Calculate total sales
region_sales = df.groupby("region")["sales"].sum().sort_values(ascending=False)
year_sales = df.groupby("year")["sales"].sum().sort_index()
category_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)


# ==========================================
# COMMON SALES RANGE
# ==========================================

all_sales_values = pd.concat([
    region_sales,
    year_sales,
    category_sales
])

low_limit = all_sales_values.quantile(1/3)
high_limit = all_sales_values.quantile(2/3)


def get_sales_color(value):

    if value < low_limit:
        return "#e74c3c"      # Red - Low

    elif value < high_limit:
        return "#f39c12"      # Orange - Medium

    else:
        return "#2ecc71"      # Green - High


# ==========================================
# ONE COLOR LEGEND FOR ALL SALES GRAPHS
# ==========================================

st.markdown(
    f"""
    ### 🎨 Sales Range

    🔴 **Low Sales:** Below ₹{low_limit:,.0f}  

    🟠 **Medium Sales:** ₹{low_limit:,.0f} – ₹{high_limit:,.0f}  

    🟢 **High Sales:** Above ₹{high_limit:,.0f}
    """
)


# ==========================================
# SALES BY REGION
# ==========================================

st.write("### 🌍 Sales by Region")

region_chart = region_sales.reset_index()

region_chart["Color"] = region_chart["sales"].apply(
    get_sales_color
)

fig_region = px.bar(
    region_chart,
    x="region",
    y="sales",
    text="sales"
)

fig_region.update_traces(
    marker_color=region_chart["Color"],
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_region.update_layout(
    xaxis_title="Region",
    yaxis_title="Total Sales (₹)",
    showlegend=False
)

st.plotly_chart(
    fig_region,
    use_container_width=True
)


# ==========================================
# SALES BY YEAR
# ==========================================

st.write("### 📅 Sales by Year")

year_chart = year_sales.reset_index()

year_chart["Color"] = year_chart["sales"].apply(
    get_sales_color
)

fig_year = px.bar(
    year_chart,
    x="year",
    y="sales",
    text="sales"
)

fig_year.update_traces(
    marker_color=year_chart["Color"],
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_year.update_layout(
    xaxis_title="Year",
    yaxis_title="Total Sales (₹)",
    showlegend=False
)

st.plotly_chart(
    fig_year,
    use_container_width=True
)


# ==========================================
# SALES BY CATEGORY
# ==========================================

st.write("### 📦 Sales by Category")

category_chart = category_sales.reset_index()

category_chart["Color"] = category_chart["sales"].apply(
    get_sales_color
)

fig_category = px.bar(
    category_chart,
    x="category",
    y="sales",
    text="sales"
)

fig_category.update_traces(
    marker_color=category_chart["Color"],
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_category.update_layout(
    xaxis_title="Category",
    yaxis_title="Total Sales (₹)",
    showlegend=False
)

st.plotly_chart(
    fig_category,
    use_container_width=True
)  

# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader("🤖 Model Comparison")

model_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [linear_r2, rf_r2]
})

fig_model = px.bar(
    model_comparison,
    x="Model",
    y="R2 Score",
    text="R2 Score"
)

fig_model.update_traces(
    marker_color=["#3498db", "#9b59b6"],
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig_model.update_layout(
    xaxis_title="Model",
    yaxis_title="R² Score",
    yaxis_range=[0, 1],
    showlegend=False
)

st.plotly_chart(
    fig_model,
    use_container_width=True
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("⭐ Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_forest_model.feature_importances_
})

fig_importance = px.bar(
    feature_importance,
    x="Feature",
    y="Importance",
    text="Importance"
)

fig_importance.update_traces(
    marker_color="#3498db",
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig_importance.update_layout(
    xaxis_title="Feature",
    yaxis_title="Importance",
    showlegend=False
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)



# ==========================================
# DATASET PREVIEW
# ==========================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)