import streamlit as st
import matplotlib.pyplot as plt

from data_pipeline import (
    load_data,
    clean_data,
    generate_kpis,
    category_wise_sales,
    region_wise_profit,
    region_wise_sales,
)
from model import train_profit_prediction_model, predict_profit


st.set_page_config(page_title="AI Business Insights Platform", layout="wide")

st.title("AI-Based Data Engineering Platform for Business Insights")
st.write(
    "This application loads business data, cleans it, generates insights, "
    "and uses machine learning to predict profit."
)

# Load Data
file_path = "business_data.csv"
data = load_data(file_path)
cleaned_data = clean_data(data)

# Dataset
st.subheader("Business Dataset")
st.dataframe(cleaned_data, use_container_width=True)

# KPIs
st.subheader("Business KPI Dashboard")
kpis = generate_kpis(cleaned_data)

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"₹{kpis['Total Sales']:,}")
col2.metric("Total Profit", f"₹{kpis['Total Profit']:,}")
col3.metric("Total Orders", kpis["Total Orders"])

col4, col5, col6 = st.columns(3)
col4.metric("Average Discount", f"{kpis['Average Discount']}%")
col5.metric("Average Sales", f"₹{kpis['Average Sales']:,}")
col6.metric("Average Profit", f"₹{kpis['Average Profit']:,}")

# Category-wise Sales Chart
st.subheader("Category-Wise Sales")
category_sales = category_wise_sales(cleaned_data)

fig1, ax1 = plt.subplots()
ax1.bar(category_sales["Product_Category"], category_sales["Sales"])
ax1.set_xlabel("Product Category")
ax1.set_ylabel("Sales")
ax1.set_title("Sales by Product Category")
st.pyplot(fig1)

# Region-wise Profit Chart
st.subheader("Region-Wise Profit")
region_profit = region_wise_profit(cleaned_data)

fig2, ax2 = plt.subplots()
ax2.bar(region_profit["Region"], region_profit["Profit"])
ax2.set_xlabel("Region")
ax2.set_ylabel("Profit")
ax2.set_title("Profit by Region")
st.pyplot(fig2)

# Region-wise Sales Chart
st.subheader("Region-Wise Sales")
region_sales = region_wise_sales(cleaned_data)

fig3, ax3 = plt.subplots()
ax3.plot(region_sales["Region"], region_sales["Sales"], marker="o")
ax3.set_xlabel("Region")
ax3.set_ylabel("Sales")
ax3.set_title("Sales by Region")
st.pyplot(fig3)

# ML Prediction
st.subheader("AI-Based Profit Prediction")
model, error = train_profit_prediction_model(cleaned_data)
st.write(f"Model Mean Absolute Error: ₹{round(error, 2)}")

sales = st.number_input("Enter Sales Amount", min_value=0, value=10000)
quantity = st.number_input("Enter Quantity", min_value=1, value=2)
discount = st.number_input("Enter Discount Percentage", min_value=0, value=5)

if st.button("Predict Profit"):
    predicted_profit = predict_profit(model, sales, quantity, discount)
    st.success(f"Predicted Profit: ₹{predicted_profit}")
