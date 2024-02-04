#!/usr/bin/env python

# Index page

Pieces = [
         ["AWS",          "EC2, t2.micro, 30 GB"],
         ["Linux",        "Ubuntu 22.04"],
         ["Python",       "High-level interpreted programming language; v3.10.12"],
         ["LangChain",    "Platform for context-aware, reasoning apps with NLP and  AI"],
         ["LlamaIndex",   "Framework for custom data index RAG and agents with NLP and AI"],
         ["Streamlit",    "Framework for web apps from pure Python"],
         ]

# Streamlit top to bottom execution

import streamlit as st

st.markdown("# ML/AI tinkering")
st.markdown("\n")
st.write(f'Some test development apps to play with in sidebar.')
st.markdown("\n")
st.write(f'Working with...')
cols0, cols1 = st.columns([.2,.8])
for i in range(0,len(Pieces)):
    cols0.write(Pieces[i][0])
    cols1.write(Pieces[i][1])

