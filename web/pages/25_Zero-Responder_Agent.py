#!/usr/bin/env python 

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_parse import LlamaParse
from operator import itemgetter
import logging
import asyncio
import nest_asyncio
import os
import streamlit as st
import sys
import threading
import time

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

emsdir="./EMSIndex"
nest_asyncio.apply()

# set API keys -- should be in env and not hardcoded here !!!
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-jOSlqgWtCVf49DysEtky0Nb8RsrVGjgNVJvy7YUqtbftrg0p"
os.environ["OPENAI_API_KEY"] = "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"

# Initialize an LLM
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="gpt-3.5-turbo",
)
llm_calm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="gpt-3.5-turbo",
)

# Define a prompt template
template = """
You are an first responder assistant helping a non-trained person 
with how to help another person (the victim) with a medical condition. 
Please first give a reassuring message and provide simple steps on how to help the victim.
"""

situational_prompt = ""

# Create conversation history store
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

output_parser = StrOutputParser()

parser = LlamaParse(
    api_key="llx-jOSlqgWtCVf49DysEtky0Nb8RsrVGjgNVJvy7YUqtbftrg0p",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True
)

#Template 1
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template + situational_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
# Compose components in a chain using LCEL
chain_calm = (
    RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
    | chat_prompt
    | llm
    | output_parser
)
#Template 2
chat_prompt_calm = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a first responder and please reassure me in this emergency situation in a single sentence and different than what you responded before"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Compose components in a chain using LCEL
chain = (
    RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
    | chat_prompt
    | llm
    | output_parser
)
chain_calm = (
    RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
    | chat_prompt_calm
    | llm_calm
    | output_parser
)

#Query function
def query_llama(input):
    return query_engine.query(input)

def query_llm(input):
    retreived_doc = query_llama(input)
    return chain.invoke({"input": retreived_doc})

# this only loads on first run, as pdf is 5MB
if 'RAG_docs' not in st.session_state:
    #st.session_state['RAG_docs'] = parser.load_data("./static/National-Model-EMS-Clinical-Guidelines_2022.pdf")
    st.session_state['RAG_docs'] = ""
    #documents = st.session_state['RAG_docs']
    #st.write('Loading "National-Model-EMS-Clinical-Guidelines_2022.pdf" to RAM; please wait a few sec...')
    #index = VectorStoreIndex.from_documents(documents)
    #Persist
    #index.storage_context.persist(persist_dir=emsdir)
    #Examine the vector_store
# Loading from disk - this is slow and every streamlit run...
st.write('RAG: Using LLAMAINDEX and reloading "National-Model-EMS-Clinical-Guidelines_2022.pdf" from disk...')
storage_context = StorageContext.from_defaults(persist_dir=emsdir)
stored_index = load_index_from_storage(storage_context)
query_engine = stored_index.as_query_engine()

# Initialize chat history
if "messy" not in st.session_state:
    st.session_state.messy = []

st.markdown("# Zero-Responder Agent")

# Display chat messages from history on app rerun
num_mess = 0
for message in st.session_state.messy:
    num_mess += 2
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if num_mess < 2:
    # Initial outbound chat prompts
    with st.chat_message("assistant", avatar="static/zero.png"):
        first_line_chat = "Welcome to Zero-Responder Agent!"
        st.markdown(first_line_chat)
    #     st.session_state.messy.append({"role": "assistant", "content": first_line_chat})
    
    with st.chat_message("assistant", avatar="static/zero.png"):
        second_line_chat = "I can help you deal with some medical emergencies."
        st.markdown(second_line_chat)
    #     st.session_state.messy.append({"role": "assistant", "content": second_line_chat})

situation_matrix=[
        ['NONE',              'No specific situation specified.'],
        ['Mobile',            'We are in a moving vehicle.'],
        ['Bloody',            'There is blood everywhere.'],
        ['Dirty',             'Situation is in a dirty soiled area.'],
        ['No tools',          'Situation is have no tools or protecctive gear.'],
        ['Suburban',          'Situation is in a suburban setting.'],
        ['Urban',             'Situation is in a urban setting.'],
        ['Rural',             'Situation is in a rural setting.'],
        ['Highway',           'Situation is in a highway setting.'],
        ['Night time',        'Situation is night time.'],
        ['Day time',          'Situation is day time.'],
#        ['Zombie Apocalypse', 'Situation is in a zombie apocalypse!']
        ]

# Input about situation
sit_name_list=[]
for items in situation_matrix:
    sit_name_list.append(items[0])
current_situation = st.sidebar.radio("What is the overall situation? (this could be replaced by real-time data)", sit_name_list)
st.sidebar.write(f"Current situation is: {current_situation}")
for items in situation_matrix:
    if items[0] == current_situation:
        situational_prompt = items[1]
st.sidebar.write(f"Current situational prompt is: {situational_prompt}")

# input about victim
with st.chat_message("assistant", avatar="static/zero.png"):
    about_victim_chat = "Please describe what is going on, what you see, what victim is saying."
    st.markdown(about_victim_chat)
# this could be augmented with take a picture of victim!!!



init_text = "OMG! Looks like my friend has a collapsed lung?!"
# if user_statement = st.text_area("description", value=init_text):
if user_statement := st.chat_input("user prompt"):
    # Display user message in chat message container
    with st.chat_message("user", avatar="static/zero_user.png"):
        st.markdown(user_statement)
    # Add user message to chat history
    st.session_state.messy.append({"role": "user", "content": user_statement})
    # Processing - what to do?
    st.sidebar.write(f"Prompt template: {template}")
    st.sidebar.write(f"Prompt situational_prompt: {situational_prompt}")
    st.sidebar.write(f"Prompt user_statement: {user_statement}")
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="static/zero.png"):
        message_placeholder = st.empty()
        response = query_llm(user_statement)
        message_placeholder = st.write(response)
        # Add assistant response to chat history
        st.session_state.messy.append({"role": "assistant", "content": response})

# Calm reminder
def async_reassuring():
    while True:
        if num_mess > 2:
            with st.chat_message("assistant", avatar="static/zero.png"):
                about_victim_chat = chain_calm.invoke({"input": "Please reassure me"})
                st.markdown(about_victim_chat)
            time.sleep(8)    

try:
    # async run the draw function, sending in all the
    # widgets it needs to use/populate
    asyncio.run(async_reassuring())
except Exception as e:
    print(f'error...{type(e)}')
    raise
finally: 
    print(f'assurance completed')
