import streamlit as st
import numpy as np
import pandas as pd
import joblib
model = joblib.load('house_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Fill in the details below to get an estimated sale price.")
st.header("Basic Info")
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
    gr_liv_area  = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
    total_bsmt   = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    first_flr    = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr   = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 0)
with col2:
    full_bath    = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    half_bath    = st.selectbox("Half Bathrooms", [0, 1, 2])
    bedroom      = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5, 6])
    tot_rms      = st.selectbox("Total Rooms Above Ground", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    garage_cars  = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
    garage_area  = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
st.header("Property Details")
col3, col4 = st.columns(2)
with col3:
    year_built   = st.number_input("Year Built", 1880, 2024, 2000)
    year_remod   = st.number_input("Year Remodelled", 1880, 2024, 2000)
    yr_sold      = st.number_input("Year Sold", 2006, 2024, 2010)
    lot_area     = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
with col4:
    lot_frontage = st.number_input("Lot Frontage (ft)", 20, 200, 70)
    mas_vnr_area = st.number_input("Masonry Veneer Area (sq ft)", 0, 1500, 0)
    bsmt_fin     = st.number_input("Finished Basement SF", 0, 2000, 0)
    open_porch   = st.number_input("Open Porch Area (sq ft)", 0, 500, 0)
log_gr_liv      = np.log1p(gr_liv_area)
log_lot_area    = np.log1p(lot_area)
total_area      = log_gr_liv + total_bsmt
total_bath      = full_bath + (0.5 * half_bath)
house_age       = yr_sold - year_built
remodel_age     = yr_sold - year_remod
rooms_per_area  = tot_rms / log_gr_liv
basement_ratio  = total_bsmt / log_gr_liv if log_gr_liv != 0 else 0
input_dict = {
    'OverallQual'   : overall_qual,
    'OverallCond'   : overall_cond,
    'GrLivArea'     : log_gr_liv,
    'TotalBsmtSF'   : total_bsmt,
    '1stFlrSF'      : first_flr,
    '2ndFlrSF'      : second_flr,
    'FullBath'      : full_bath,
    'HalfBath'      : half_bath,
    'BedroomAbvGr'  : bedroom,
    'TotRmsAbvGrd'  : tot_rms,
    'GarageCars'    : garage_cars,
    'GarageArea'    : garage_area,
    'YearBuilt'     : year_built,
    'YearRemodAdd'  : year_remod,
    'YrSold'        : yr_sold,
    'LotArea'       : log_lot_area,
    'LotFrontage'   : lot_frontage,
    'MasVnrArea'    : mas_vnr_area,
    'BsmtFinSF1'    : bsmt_fin,
    'OpenPorchSF'   : open_porch,
    'TotalArea'     : total_area,
    'TotalBathrooms': total_bath,
    'HouseAge'      : house_age,
    'RemodelAge'    : remodel_age,
    'RoomsPerArea'  : rooms_per_area,
    'BasementRatio' : basement_ratio,
}
input_df = pd.DataFrame([input_dict])
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]
if st.button("Predict Sale Price"):
    log_prediction = model.predict(input_df)[0]
    actual_price   = np.expm1(log_prediction)
    st.success(f"### Estimated Sale Price: ${actual_price:,.0f}")
    st.caption("Prediction made using Linear Regression trained on Ames Housing dataset.")
