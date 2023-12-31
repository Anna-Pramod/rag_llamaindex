import streamlit as st
from streamlit_chat import message

import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import requests
session = requests.Session()
session.verify = False
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY']='http://gateway.zscaler.net:9400'
os.environ['HTTPS_PROXY']='http://gateway.zscaler.net:9400'

documents = SimpleDirectoryReader("./appa/Purchase procedure dated 12.04.2023.pdf").load_data()

# required_exts = [".md"]
# reader = SimpleDirectoryReader(
#     input_dir="../../end_to_end_tutorials",
#     required_exts=required_exts,
#     recursive=True,
# )
# documents = reader.load_data()

llm = LlamaCPP(
    # model_url='https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf',
    # model_path="/mnt/c/Users/235095/Documents/tstchat/LLama2ChatBot/llama-2-7b.Q2_K.gguf",
    model_path="/mnt/c/Users/235095/Documents/tstchat/LLama2ChatBot/Nous-Capybara-3B-V1.9.Q8_K.gguf",
    temperature=0,
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="/mnt/c/Users/235095/Documents/tstchat/LLama2ChatBot/all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model,
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

st.set_page_config(page_title="ChatBot 🧑🏽 (CPU)", page_icon=":guardsman:", layout="wide")
st.title("ChatBot 🧑🏽 - CPU version")

def conversation_chat(query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    res = response.response
    st.session_state['history'].append((query, res))
    return (res)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
 
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=False):
            user_input = st.text_input("Question:", placeholder="Shoot!", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

 

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()
