import streamlit as st
import os

st.set_page_config(page_title='Pneumonia Detection | Notebook', page_icon="🤕", layout="centered", initial_sidebar_state="auto", menu_items=None)

with open('./pages/chest-xray-pneumonia-normal-binary-classification.html', "r", encoding="utf-8") as file:
    html_content = file.read()

st.html(html_content)