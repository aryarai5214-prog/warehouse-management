import streamlit as st
from database import init_db, get_products, record_stock

st.set_page_config(page_title="Stock Out", page_icon="📤", layout="wide")
init_db()
st.title("📤 Stock Out")
st.write("Record items dispatched from the warehouse.")

products = get_products()
if products.empty:
    st.info("Add a product first on the Products page.")
else:
    with st.form("stock_out"):
        sku = st.selectbox("Product", products["sku"].tolist(),
                           format_func=lambda x: f"{x} — {products.loc[products['sku'] == x, 'name'].iloc[0]} (available: {int(products.loc[products['sku'] == x, 'quantity'].iloc[0])})")
        qty = st.number_input("Quantity Dispatched", min_value=1, step=1)
        note = st.text_input("Order / Dispatch Reference")
        submitted = st.form_submit_button("Record Stock Out", type="primary")
        if submitted:
            try:
                record_stock(sku, "OUT", qty, note)
                st.success("Stock-out transaction recorded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
