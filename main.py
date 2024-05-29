import streamlit as st
st.set_page_config(page_title='Knop Bot Chat',
                   initial_sidebar_state='collapsed')
from openai import OpenAI
from langchain_utils import invoke_chain
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import random
import time
import random

def reset_chat_history():
    if "messages" in st.session_state:
        st.session_state.messages = []
        
model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
max_tokens = {
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "gemma-7b-it": 8192
}
# Initialize model
if "model" not in st.session_state:
    print("Selecting initial model")
    st.session_state.model = model_options[0]
    st.session_state.temperature = 0
    st.session_state.max_tokens = 8192

# Initialize chat history
if "messages" not in st.session_state:
    print("Creating session state...")
    st.session_state.messages = []
    #print(f"Session State -> {st.session_state}")
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '''<div class="center">
    <img src="https://image.cdn2.seaart.ai/2023-08-29/15666278314310661/0311de0e5c123032e9f78232d8ced9182cc870b6_low.webp" width="150" style="border-radius: 50%; display: block; margin: auto;">
</div>

<style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
''',
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center;'>Knop Bot</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Función para renderizar o actualizar el contenido del markdown
def render_or_update_model_info(model):
    st.markdown(
        '<div class="center">' +
        f'Este chatbot utiliza&nbsp;<b>{model}</b>' +
        '</div>',
        unsafe_allow_html=True
    )


with st.sidebar:
    st.title("Configuración de modelo")

    st.session_state.model = st.selectbox(
        "Elige un modelo:",
        model_options,
        index=0
    )

    st.session_state.temperature = st.slider('Selecciona una temperatura:', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")


    if st.session_state.max_tokens > max_tokens[st.session_state.model]:
        max_value = max_tokens[st.session_state.model]


    st.session_state.max_tokens = st.number_input('Seleccione un máximo de tokens:', min_value=1, max_value=max_tokens[st.session_state.model], value=max_tokens[st.session_state.model], step=100)

    if st.button("Vaciar Chat"):
        reset_chat_history()


render_or_update_model_info(st.session_state.model)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "figure" in message["aux"].keys() and len(message["aux"]["figure"]) > 0:
            st.plotly_chart(message["aux"]["figure"][0])
        st.text("")
# Accept user input
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        aux_model = random.choice(model_options)

        response = invoke_chain(question=prompt, 
                                messages=st.session_state.messages, 
                                model_name=model_options[model_options.index(st.session_state.model)],
                                temperature=st.session_state.temperature,
                                max_tokens=st.session_state.max_tokens
                                )
        st.write_stream(response)
        if "figure" in invoke_chain.aux.keys() and len(invoke_chain.aux["figure"]) > 0:
            st.plotly_chart(invoke_chain.aux["figure"][0])
        if hasattr(invoke_chain, 'recursos'):
            for recurso in invoke_chain.recursos:
                st.button(recurso)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "aux": {}})
    st.session_state.messages.append({"role": "assistant", "content": invoke_chain.response, "aux": invoke_chain.aux})