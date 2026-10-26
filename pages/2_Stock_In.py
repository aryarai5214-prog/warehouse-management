import streamlit as st
from database import init_db, get_products, record_stock

st.set_page_config(page_title="Stock In", page_icon="📥", layout="wide")
init_db()
st.title("📥 Stock In")
st.write("Record goods received into the warehouse.")

products = get_products()
if products.empty:
    st.info("Add a product first on the Products page.")
else:
    with st.form("stock_in"):
        sku = st.selectbox("Product", products["sku"].tolist(),
                           format_func=lambda x: f"{x} — {products.loc[products['sku'] == x, 'name'].iloc[0]}")
        qty = st.number_input("Quantity Received", min_value=1, step=1)
        note = st.text_input("Reference / Note", placeholder="e.g. supplier delivery or purchase order")
        submitted = st.form_submit_button("Record Stock In", type="primary")
        if submitted:
            try:
                record_stock(sku, "IN", qty, note)
                st.success("Stock-in transaction recorded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
