import pandas as pd


def load_data(file_path):
    """Load business data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """Clean duplicate and missing values from the dataset."""
    data = data.drop_duplicates()

    data["Sales"] = data["Sales"].fillna(data["Sales"].mean())
    data["Profit"] = data["Profit"].fillna(data["Profit"].mean())
    data["Quantity"] = data["Quantity"].fillna(data["Quantity"].median())
    data["Discount"] = data["Discount"].fillna(0)

    return data


def generate_kpis(data):
    """Generate important business KPIs."""
    return {
        "Total Sales": data["Sales"].sum(),
        "Total Profit": data["Profit"].sum(),
        "Total Orders": data["Order_ID"].nunique(),
        "Average Discount": round(data["Discount"].mean(), 2),
        "Average Sales": round(data["Sales"].mean(), 2),
        "Average Profit": round(data["Profit"].mean(), 2),
    }


def category_wise_sales(data):
    """Calculate total sales by product category."""
    return data.groupby("Product_Category", as_index=False)["Sales"].sum()


def region_wise_profit(data):
    """Calculate total profit by region."""
    return data.groupby("Region", as_index=False)["Profit"].sum()


def region_wise_sales(data):
    """Calculate total sales by region."""
    return data.groupby("Region", as_index=False)["Sales"].sum()
