#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow_hub as hub
import os
import streamlit as st
from PIL import Image
from collections import OrderedDict


def image_to_tensor(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [720, 512])
    img = img[tf.newaxis, :]
    return img

@st.cache(suppress_st_warning=True)
def load_model_stylization():
    mod_style = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return mod_style


@st.cache(suppress_st_warning=True)
def styles_and_content():
    style_path = r'./styles/'
    file_extention = '.jpg'
    style_name1 = 'Van Gogh'
    style_name2 = 'Leonardo'
    style_name3 = 'Munch'
    style_name4 = 'Hokusai'
    style_name5 = 'Kandinsky'
    return style_path, file_extention, style_name1, style_name2, style_name3, style_name4, style_name5


@st.cache(suppress_st_warning=True)
def load_style():
    
    st_path, file_ext, st_name1, st_name2, st_name3, st_name4, st_name5 = styles_and_content()
    
    style_1 = image_to_tensor(st_path + st_name1 + file_ext)
    style_2 = image_to_tensor(st_path + st_name2 + file_ext)
    style_3 = image_to_tensor(st_path + st_name3 + file_ext)
    style_4 = image_to_tensor(st_path + st_name4 + file_ext)
    style_5 = image_to_tensor(st_path + st_name5 + file_ext)
    ord_dict = OrderedDict()
    return style_1, style_2, style_3, style_4, style_5, ord_dict, st_name1, st_name2, st_name3, st_name4, st_name5


@st.cache(suppress_st_warning=True, allow_output_mutation=True) 
def make_dict():
    
    st1,st2,st3,st4,st5,ordict,sty_name1,sty_name2,sty_name3,sty_name4,sty_name5 = load_style()
    ordict = {sty_name1:st1, sty_name2:st2, sty_name3:st3, sty_name4:st4, sty_name5:st5}
    return ordict


add_selectbox = st.sidebar.selectbox('Which style do you prefer?',list(make_dict().keys()) )

st.sidebar.write("You chose: " + add_selectbox)


@st.cache(suppress_st_warning=True)
def show_style(add_sbox):
    diction = make_dict()
    style_pic = diction[add_sbox]
    style_pic = np.array(style_pic*255, dtype=np.uint8)[0]
    return style_pic

style_to_show = show_style(add_selectbox)

st.sidebar.image(style_to_show, caption=add_selectbox, use_column_width=True)


@st.cache(suppress_st_warning=True)
def load_content():
    content_path = r'./pics/fisherman.png'
    content_image_tensor = image_to_tensor(content_path)
    return content_image_tensor

content_image_tensor = load_content()

@st.cache(suppress_st_warning=True)
def model_run(sbox):
    dic = make_dict()
    cont_image = load_content()
    style_image = dic[sbox]
    module = load_model_stylization()
    combined_result = module(tf.constant(cont_image), tf.constant(style_image))[0]
    return combined_result

combined_result = model_run(add_selectbox)

st.title('Turning pictures into paintings')
st.write('A Python implementation')
                  
              
tensor = combined_result*255
tensor = np.array(tensor, dtype=np.uint8)
tensor = tensor[0]

st.image(tensor, caption='Blended result', use_column_width=True)
st.write('More on Instagram at "Pictures_made_paintings"')
