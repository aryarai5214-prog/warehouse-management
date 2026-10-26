import streamlit as st
import pandas as pd
import plotly.express as px
from database import init_db, get_products, get_transactions, add_product, record_stock

st.set_page_config(page_title="Warehouse Management System", page_icon="📦", layout="wide")

init_db()

st.title("📦 Warehouse Management System")
st.caption("Inventory, stock movement, and reporting")

products = get_products()
transactions = get_transactions()

if products.empty:
    total_products = 0
    total_units = 0
    low_stock_count = 0
    inventory_value = 0
else:
    total_products = len(products)
    total_units = int(products["quantity"].sum())
    low_stock_count = int((products["quantity"] <= products["reorder_level"]).sum())
    inventory_value = float((products["quantity"] * products["unit_price"]).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products", total_products)
c2.metric("Units in Stock", f"{total_units:,}")
c3.metric("Low-Stock Items", low_stock_count)
c4.metric("Inventory Value", f"₹{inventory_value:,.2f}")

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("Current Inventory")
    if products.empty:
        st.info("No products yet. Add products from the Products page.")
    else:
        display = products.copy()
        display["unit_price"] = display["unit_price"].map(lambda x: f"₹{x:,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

with right:
    st.subheader("Stock by Product")
    if products.empty:
        st.info("Add products to see the chart.")
    else:
        fig = px.bar(products, x="name", y="quantity", color="category",
                     labels={"name": "Product", "quantity": "Quantity", "category": "Category"})
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Low-Stock Alerts")
if products.empty:
    st.info("No inventory data available yet.")
else:
    low = products[products["quantity"] <= products["reorder_level"]]
    if low.empty:
        st.success("No products are currently at or below their reorder level.")
    else:
        st.warning(f"{len(low)} product(s) need attention.")
        st.dataframe(low[["sku", "name", "quantity", "reorder_level"]], use_container_width=True, hide_index=True)

st.subheader("Recent Stock Movements")
if transactions.empty:
    st.info("Stock movements will appear here after you record stock in or stock out.")
else:
    st.dataframe(transactions.head(10), use_container_width=True, hide_index=True)
