import time
import numpy as np  
import pandas as pd  
from PIL import Image
from sqlalchemy import create_engine,insert
import plotly.express as px 
import plotly.graph_objects as go
import streamlit as st  
import plotly.express as px
import altair as alt
import logging
import os
import requests
import os
import random

# request image batch
def image_request(files):
    url = "http://127.0.0.1:5000/inference"
    headers = {"accept": "application/json"}

    response = requests.post(url, headers=headers, files=files)
    return response.json()

# prepare engine
def prepare_db(db):
    engine = create_engine(db)
    return engine

# get data
def get_data(engine, shift_type, window_size):
    assert shift_type in ['cov_shift','label_shift','model_drift']
    df = pd.read_sql_query(f'SELECT * FROM {shift_type}',engine)
    df['average'] = df['score'].rolling(window_size).mean()
    return df

def interval(cnt,engine,image_list):
    shifted = image_list[0].split('@')[1]
    cov_df = get_data(engine, 'cov_shift',5)
    label_df = get_data(engine, 'label_shift', 5)
    model_df = get_data(engine, 'model_drift', 5)
    fig_col11, fig_col12 = st.columns([3,1])
    with fig_col11:
        st.markdown("##### Covariate Shift")
        score_df = cov_df[['score']]
        score_df['label'] = 'score'
        min_axis=0.9*min(score_df['score'])
        shift_df = pd.DataFrame(cov_df.apply(lambda x: x['score'] if x['detected'] == True else None,axis=1))
        shift_df['label'] = 'shifted'
        shift_df['tmp_size'] = 1
        shift_df = shift_df.rename(columns={0:'score'})        
        average_df = cov_df[['average']]
        average_df['label'] = 'average'
        average_df = average_df.rename(columns = {'average':'score'})
        new_df = pd.concat([score_df,average_df],join='inner')
        new_df = new_df.reset_index()
        
        fig = px.scatter(shift_df.reset_index(), x='index', y='score', size='tmp_size', size_max=15)
        fig.add_trace(
            go.Scatter(
                x=score_df.reset_index()['index'],
                y=score_df['score'],
                mode='lines',
                name='score'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=average_df.reset_index()['index'],
                y=average_df['score'],
                mode='lines',
                name='average'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        # c = alt.Chart(new_df).mark_line().encode(
        #     x=alt.X('index'),
        #     y=alt.Y('score', scale=alt.Scale(domain=[0.9*min(score_df['score']), 1.1*max(score_df['score'])])),
        #     color='label',
        # ).properties(height=200)
        # st.altair_chart(c, use_container_width=True)
        # st.line_chart(cov_df[['score','average']],height=150)
    with fig_col12:
        score_value = format(cov_df.iloc[-1]['score'], ".3f")
        average_value = format(cov_df.iloc[-1]['average'], ".3f")
        if shifted == "normal":
            score_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{average_value}</p>'
        else:
            score_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{average_value}</p>'
        st.write("Current Covariate Shift")
        st.markdown(score_title, unsafe_allow_html=True)
        st.write("Average Covariate Shift")
        st.markdown(average_title, unsafe_allow_html=True)

    fig_col21, fig_col22 = st.columns([3,1])
    with fig_col21:
        st.markdown("##### Label Shift")
        score_df = label_df[['score']]
        score_df['label'] = 'score'
        min_axis=0.9*min(score_df['score'])
        shift_df = pd.DataFrame(label_df.apply(lambda x: x['score'] if x['detected'] == True else None,axis=1))
        shift_df['label'] = 'shifted'
        shift_df['tmp_size'] = 1
        shift_df = shift_df.rename(columns={0:'score'})
        average_df = label_df[['average']]
        average_df['label'] = 'average'
        average_df = average_df.rename(columns = {'average':'score'})

        new_df = pd.concat([score_df,average_df],join='inner')
        new_df = new_df.reset_index()
        
        # fig = px.bar(shift_df.reset_index(), x='index', y='score')
        fig = px.scatter(shift_df.reset_index(), x='index', y='score', size='tmp_size', size_max=15)
        fig.add_trace(
            go.Scatter(
                x=score_df.reset_index()['index'],
                y=score_df['score'],
                mode='lines',
                name='score'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=average_df.reset_index()['index'],
                y=average_df['score'],
                mode='lines',
                name='average'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        # c = alt.Chart(new_df).mark_line().encode(
        #     x=alt.X('index'),
        #     y=alt.Y('score', scale=alt.Scale(domain=[0.9*min(score_df['score']), 1.1*max(score_df['score'])])),
        #     color='label',
        # ).properties(height=200)

        # st.altair_chart(c, use_container_width=True)
    with fig_col22:
        score_value = format(label_df.iloc[-1]['score'], ".3f")
        average_value = format(label_df.iloc[-1]['average'], ".3f")
        if shifted == "normal":
            score_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{average_value}</p>'
        else:
            score_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{average_value}</p>'
        st.write("Current Label Shift")
        st.markdown(score_title, unsafe_allow_html=True)
        st.write("Average Label Shift")
        st.markdown(average_title, unsafe_allow_html=True)

    fig_col31, fig_col32 = st.columns([3,1])
    with fig_col31:
        st.markdown("##### Model Drift")
        score_df = model_df[['score']]
        score_df['label'] = 'score'
        min_axis=0.9*min(score_df['score'])
        shift_df = pd.DataFrame(model_df.apply(lambda x: x['score'] if x['detected'] == True else None,axis=1))
        shift_df['label'] = 'shifted'
        shift_df['tmp_size'] = 1
        shift_df = shift_df.rename(columns={0:'score'})
        average_df = model_df[['average']]
        average_df['label'] = 'average'
        average_df = average_df.rename(columns = {'average':'score'})
        new_df = pd.concat([score_df,average_df],join='inner')
        new_df = new_df.reset_index()
        
        fig = px.scatter(shift_df.reset_index(), x='index', y='score', size='tmp_size', size_max=15)
        fig.add_trace(
            go.Scatter(
                x=score_df.reset_index()['index'],
                y=score_df['score'],
                mode='lines',
                name='score'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=average_df.reset_index()['index'],
                y=average_df['score'],
                mode='lines',
                name='average'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        # c = alt.Chart(new_df).mark_line().encode(
        #     x=alt.X('index'),
        #     y=alt.Y('score', scale=alt.Scale(domain=[0.9*min(score_df['score']), 1.1*max(score_df['score'])])),
        #     color='label',
        # ).properties(height=200)
        # st.altair_chart(c, use_container_width=True)
    with fig_col32:
        score_value = format(model_df.iloc[-1]['score'], ".3f")
        average_value = format(model_df.iloc[-1]['average'], ".3f")
        if shifted == "normal":
            score_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">{average_value}</p>'
        else:
            score_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{score_value}</p>'
            average_title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;">{average_value}</p>'
        st.write("Current Model Drift")
        st.markdown(score_title, unsafe_allow_html=True)
        st.write("Average Model Drift")
        st.markdown(average_title, unsafe_allow_html=True)

    # sidebar = st.sidebar.empty()
    # with st.sidebar:
    if shifted == "label":
        fig_col41, fig_col42 = st.columns([1,1])
        with fig_col41:
            normal_df = label_df.iloc[:10][['count0','count1','count2','count3']]
            normal_data = pd.DataFrame(
                {'count':[normal_df['count0'].sum(), normal_df['count1'].sum(), normal_df['count2'].sum(), normal_df['count3'].sum()]}
            )
            st.bar_chart(normal_data)
            st.markdown('<p style="font-family:sans-serif; color:Gray; text-align:center; font-size: 15px;">Groud Truth</p>', unsafe_allow_html=True)

        with fig_col42:
            x,class_count = np.unique([i.split('@')[2]for i in image_list], return_counts=True)
            # with fig_col42:
            shifted_data = pd.DataFrame(
                {'count': class_count}
            )
            st.bar_chart(shifted_data)
            st.markdown('<p style="font-family:sans-serif; color:Gray; text-align:center; font-size: 15px;">Shifted</p>', unsafe_allow_html=True)
    
    if shifted == "corr":
        fig_col41 = st.columns([1])
        # with fig_col41:
        image_root = "../data/inference_n"
        image_list_n = sorted([os.path.join(image_root, i) for i in os.listdir(image_root)],
                            key=lambda x: int(x.split('@')[0].split('/')[-1]))

        img_base = Image.new("RGB", (500, 50), "white")
        for i in range(10):
            im = Image.open(image_list_n[i]).resize((50,50))
            img_base.paste(im, (i*50,0))

        st.image(img_base,width=4000,caption='Ground Truth')

        # with fig_col42:
        image_list_tmp = sorted(image_list, key=lambda x: random.random())
        img_base = Image.new("RGB", (500, 50), "white")
        for i in range(10):
            im = Image.open(image_list_tmp[i]).resize((50,50))
            img_base.paste(im, (i*50,0))

        st.image(img_base,width=4000,caption='Corrupted')
    return cnt