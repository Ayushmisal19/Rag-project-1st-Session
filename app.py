
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions clearly."),
        ("human", "Question: {question}")
    ]
)

# Streamlit framework
st.title("LangChain Demo with Gemma 2B")
input_text = st.text_input("What question do you have in mind?")

# Ollama model
llm = OllamaLLM(model="gemma2:2b")
output_parser = StrOutputParser()

# Create chain
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)

