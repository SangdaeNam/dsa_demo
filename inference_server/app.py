import time
import numpy as np  
import pandas as pd  
from PIL import Image
import sqlite3
import altair as alt
from sqlalchemy import create_engine,insert
from utils import prepare_db,get_data,interval,image_request
import plotly.express as px 
import streamlit as st  
from streamlit import caching
from multiprocessing import Process
import logging
import os
import psutil   
import random

logging.basicConfig(filename = 'streamlit.log')

st.set_page_config(
    page_title="DSA Demo",
    page_icon="✅",
    layout="wide",
)

image_root = "../data/inference_c_new"
db = 'sqlite:///../data/dsa_test.db'
image_list = sorted([os.path.join(image_root, i) for i in os.listdir(image_root)],
                    key=lambda x: int(x.split('@')[0].split('/')[-1]),reverse=True)
batch_size = 128
batch_end = int(len(image_list)/batch_size)

engine = prepare_db(db)

# dashboard title
st.title("DSA Demo")

fig_bot1, fig_bot2, _ = st.columns([1,1,25])
with fig_bot1:
    start = st.checkbox("Start")
with fig_bot2:
    reset = st.button('Reset')
if reset:
    engine.execute("DELETE from cov_shift where shifted='shifted'")
    engine.execute("DELETE from label_shift where shifted='shifted'")
    engine.execute("DELETE from model_drift where shifted='shifted'")
    cnt = 0
    cnt = interval(cnt,engine,image_list)   

cnt = 0

placeholder = st.empty()    
with placeholder.container():
    _ = interval(cnt,engine,image_list)

if start:
    while True:
        with placeholder.container():
            if cnt == batch_end:
                image_batch_list = image_list[cnt*batch_size:]
            else:
                image_batch_list = image_list[cnt*batch_size:(cnt+1)*batch_size]
            
            files = {}
            for i,v in enumerate(image_batch_list):
                key = f'img{i}'
                files[key] = open(v, "rb")
            
            response = image_request(files)
            cnt = interval(cnt,engine,image_batch_list)

            if start != True:
                break
            if cnt == batch_end:
                break
            else:
                cnt += 1
                time.sleep(2)