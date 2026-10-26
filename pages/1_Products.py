import streamlit as st
from database import init_db, get_products, add_product, delete_product

st.set_page_config(page_title="Products", page_icon="📦", layout="wide")
init_db()
st.title("📦 Product Management")

with st.form("add_product", clear_on_submit=True):
    st.subheader("Add a Product")
    c1, c2 = st.columns(2)
    sku = c1.text_input("SKU / Product Code *")
    name = c2.text_input("Product Name *")
    category = c1.text_input("Category", value="General")
    location = c2.text_input("Storage Location", placeholder="e.g. Rack A-01")
    c3, c4, c5 = st.columns(3)
    quantity = c3.number_input("Opening Quantity", min_value=0, step=1)
    reorder_level = c4.number_input("Reorder Level", min_value=0, value=10, step=1)
    unit_price = c5.number_input("Unit Price (₹)", min_value=0.0, step=10.0)
    submitted = st.form_submit_button("Add Product", type="primary")
    if submitted:
        if not sku.strip() or not name.strip():
            st.error("SKU and Product Name are required.")
        else:
            try:
                add_product(sku, name, category, quantity, reorder_level, unit_price, location)
                st.success("Product added successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add product: {exc}")

st.divider()
st.subheader("Search and View Products")
products = get_products()
query = st.text_input("Search by SKU, name, category, or location")
if not products.empty and query:
    mask = products.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
    products = products[mask]
st.dataframe(products, use_container_width=True, hide_index=True)

st.subheader("Delete a Product")
st.caption("Deleting a product removes it from the product list but keeps its historical stock transactions.")
all_products = get_products()
if all_products.empty:
    st.info("There are no products to delete.")
else:
    sku_to_delete = st.selectbox("Select SKU", all_products["sku"].tolist())
    if st.button("Delete Selected Product"):
        try:
            delete_product(sku_to_delete)
            st.success(f"Deleted product {sku_to_delete}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not delete product: {exc}")
