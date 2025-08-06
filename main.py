import streamlit as st

import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
ndf = pickle.load(open('ndf.pkl', 'rb'))

st.title('Laptop Price Predictor')


company = st.selectbox('Company',ndf['Company'].unique())
type = st.selectbox('Type',ndf['TypeName'].unique())
inches = st.number_input('Screen Size')
ram = st.selectbox('RAM',[2,4,6,8,12,16,24,32,64])
op = st.selectbox('OS',ndf['OpSys'].unique())

weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen',['Yes','No'])
ips = st.selectbox('IPS',['Yes','No'])
processor = st.selectbox('Processor',ndf['Processor'].unique())
ssd = st.selectbox('SSD',[0,8,128,256,512,1024, 2048,3072])
hdd = st.selectbox('HDD',[0,128,256,512,1024,2048,3072])
flash = st.selectbox('Flash Storage (GB)',[0,128,256,512,1024,2048,3072])
gpu = st.selectbox('GPU',ndf['Graphics'].unique())
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])



if st.button('Predict Price'):
    # query
    displayArea = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    height = int(resolution.split('x')[0])
    width = int(resolution.split('x')[1])
    displayArea = height * width
    query = np.array([company,type,inches,ram,op,weight,touchscreen,ips,processor,ssd,hdd,flash,gpu,displayArea])

    query = query.reshape(1,14)
    st.title(np.exp(pipe.predict(query)))
   