# Warehouse Management System (Streamlit)

A starter warehouse inventory application built with Python, Streamlit, Pandas, Plotly, and SQLite.

## Features
- Dashboard with product count, total units, inventory value, and low-stock alerts
- Add and search products
- Record stock-in and stock-out movements
- Prevent stock-out transactions that exceed available stock
- View stock movement history
- Download inventory and transaction reports as CSV

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io/
3. Create an app from your repository and select `app.py`.
4. Deploy.

## Important deployment note
This starter uses a local SQLite database file (`warehouse.db`). Streamlit Community Cloud's local filesystem is not guaranteed to persist across redeployments or restarts. For real production or multi-user use, switch to a managed persistent database (such as hosted PostgreSQL or MySQL) before relying on it for business inventory.

## Current limitations
- No authentication or role-based permissions yet.
- Supplier page is a placeholder.
- This is a starter project; review and test before using it for real warehouse operations.
