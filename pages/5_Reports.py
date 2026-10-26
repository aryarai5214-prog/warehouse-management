import streamlit as st
import pandas as pd
from database import init_db, get_products, get_transactions

st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")
init_db()
st.title("📊 Reports")

products = get_products()
transactions = get_transactions()

st.subheader("Inventory Report")
if products.empty:
    st.info("No products available to report.")
else:
    report = products.copy()
    report["inventory_value"] = report["quantity"] * report["unit_price"]
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Inventory CSV",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="warehouse_inventory_report.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Stock Movement History")
if transactions.empty:
    st.info("No stock movements recorded yet.")
else:
    st.dataframe(transactions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Transactions CSV",
        data=transactions.to_csv(index=False).encode("utf-8"),
        file_name="warehouse_stock_transactions.csv",
        mime="text/csv"
    )
