import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from roboflow import Roboflow
import tensorflow as tf
import cv2
# Page title
st.set_page_config(page_title='lemonQC', page_icon='🍋')
st.title('🍋x🤖 Smart Lemon Insight')
st.divider()

with st.expander('***About this app***'):
  st.info('*Smart Lemon Insight adalah sebuah sistem kecerdasan buatan yang dirancang untuk menentukan kualitas buah lemon secara otomatis dan presisi. Sistem ini menggunakan teknologi deep learning dengan memanfaatkan Keras dari TensorFldow dan algoritma YOLO (You Only Look Once) untuk melatih model deteksi objek. Dengan KerasCV, Smart Lemon Insight dapat memanfaatkan model YOLOv8 yang telah dilatih sebelumnya untuk mendeteksi dan mengklasifikasikan lemon berdasarkan parameter seperti warna, ukuran, dan ada tidaknya cacat. Proses ini melibatkan pengolahan citra digital yang mencakup augmentasi data, konversi ruang warna, dan ekstraksi fitur. Dengan metode ini, Smart Lemon Insight mampu memberikan penilaian kualitas lemon secara real-time dengan akurasi tinggi, sehingga membantu petani dan produsen lemon dalam memastikan kualitas produk mereka tetap konsisten dan optimal*')
  
  st.markdown('**What can this app do?**')
  st.info('Smart Lemon Insight mampu mengklasifikasikan lemon ke dalam beberapa kategori mutu berdasarkan parameter visual seperti warna dan ukuran, memastikan bahwa hanya lemon berkualitas tinggi yang dipasarkan.')

  st.markdown('**Data sets:**')
  st.code('''- https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset
  ''', language='markdown')
  


