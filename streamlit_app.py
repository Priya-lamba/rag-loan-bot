import streamlit as st
from main import ask

st.title(" Loan Approval Chatbot (RAG)")
query = st.text_input("Ask me anything about the loan dataset")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            response = ask(query)
            st.success(response)
