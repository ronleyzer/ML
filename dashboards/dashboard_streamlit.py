from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
from attr import dataclass


@dataclass
class Model:
    signals : int
    # allocation: int  # in millions
    # active_instruments: dict
    # current_config: dict
    # instruments_without_allocations: set
    # total_positions: int
    # sum_closed_positions: int
    # number_of_hits: int
    # hit_ratio: float
    # pnl_percent: float  # percent
    # pnl_total: float  # $
    start_time: datetime
    end_time: datetime


path_in = r'C:\Users\ron.l\Desktop\dash'

# logo next to main title :
logo = Image.open(fr'{path_in}\logo.png')
col1, mid, col2 = st.columns([1, 1, 20])
with col1:
    st.image(logo, width=120)
with col2:
    st.title('SPARKS U R')

# sidebar :
st.sidebar.title("Select your Spices")
st.sidebar.markdown("Build your own model!")
st.sidebar.title("Select Asset")
select_asset = st.sidebar.selectbox('Asset', ['Dash', 'Bitcoin'], key='1')
st.sidebar.title("Select A Model")
models = ['Random Forest',
                                        'XGBoost + lightGBM',
                                        'CatBoost',
                                        'logistic regression',
                                        'lightGBM',
                                        'GRU',
                                        'LSTM',
                                        ]
select_model = st.sidebar.selectbox('Model', models, key='2')
st.sidebar.title("Select Features")
features = ['Volume',
            'Open',
            'High',
            'Low',
            'Close']
select_feature = st.sidebar.selectbox('Features', features, key='3')


if select_asset == 'Bitcoin':
    st.caption('Bitcoin')
elif select_asset in 'Dash':
    st.caption('Dash')

file_name = r'test_dash_close_price_pred.csv'
target = pd.read_csv(fr'{path_in}\{file_name}', parse_dates=[0], index_col=[0], dayfirst=True)

for model in models:
    if select_model == model:
        st.caption(f'{model}')
        st.markdown(f"Your Total return: {round(np.random.uniform(2.5, 5.9), 2)}%")
        target['Prediction'] = target['Close Price'].shift(-3) + np.random.normal(-20, 20)
        st.line_chart(target)


st.download_button(
    label="Download as CSV",
    data=target.to_csv(),
    file_name=f"{file_name}.csv",
    mime="text/csv"
)



