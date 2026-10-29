
import os
import tempfile
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- STREAMLIT UI ----------------

st.set_page_config(
    page_title="RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.write(
    "Upload your documents and ask questions. "
    "Answers will appear directly on this page."
)


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.header("Settings")

model_name = st.sidebar.text_input(
    "Ollama model",
    value="gemma2:2b"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=200,
    max_value=1500,
    value=500,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=300,
    value=50,
    step=10
)

num_chunks = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=8,
    value=4
)


# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "Upload TXT, PDF or DOCX files",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)


# ---------------- LOAD DOCUMENTS ----------------

def load_document(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        if suffix == ".pdf":
            documents = PyPDFLoader(temp_path).load()

        elif suffix == ".docx":
            documents = Docx2txtLoader(temp_path).load()

        elif suffix == ".txt":
            documents = TextLoader(
                temp_path,
                encoding="utf-8"
            ).load()

        else:
            raise ValueError("Unsupported file format")

        # Save original filename as metadata
        for document in documents:
            document.metadata["source"] = uploaded_file.name

        return documents

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- SESSION STATE ----------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------- BUILD KNOWLEDGE BASE ----------------

if st.button("Build Knowledge Base", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        try:
            with st.spinner(
                "Loading documents and creating embeddings..."
            ):

                documents = []

                for uploaded_file in uploaded_files:
                    documents.extend(
                        load_document(uploaded_file)
                    )

                if not documents:
                    st.error("No readable content found.")
                    st.stop()

                # Split text into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=min(
                        chunk_overlap,
                        chunk_size - 1
                    )
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    st.error("No text chunks were created.")
                    st.stop()

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=(
                        "sentence-transformers/"
                        "all-MiniLM-L6-v2"
                    )
                )

                # Store vectors in FAISS
                vector_db = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                # Update only after successful creation
                st.session_state.vector_db = vector_db

                st.session_state.indexed_files = [
                    file.name for file in uploaded_files
                ]

            st.success(
                f"Knowledge base created successfully! "
                f"{len(chunks)} text chunks indexed."
            )

        except Exception as e:
            st.error(f"Error building knowledge base: {e}")


# ---------------- DISPLAY KNOWLEDGE BASE STATUS ----------------

if st.session_state.vector_db is not None:

    st.success("Your knowledge base is ready.")

    st.caption(
        "Indexed files: "
        + ", ".join(st.session_state.indexed_files)
    )

    st.divider()

    # ---------------- QUESTION INPUT ----------------

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="What is the main topic of the document?"
    )

    ask_button = st.button("Generate Answer", type="primary")

    if ask_button:

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching documents and generating your answer..."
                ):

                    # Retrieve relevant text chunks
                    relevant_docs = (
                        st.session_state.vector_db
                        .similarity_search(
                            question,
                            k=num_chunks
                        )
                    )

                    # Combine retrieved text
                    context = "\n\n".join(
                        document.page_content
                        for document in relevant_docs
                    )

                    # Create prompt template
                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """
You are a helpful question-answering assistant.

Answer the user's question using the supplied context.

Rules:
1. Return the answer directly as readable text.
2. Do not open, link to, or return a document.
3. Do not return file paths.
4. Do not invent facts that are absent from the context.
5. If the answer is not available in the context,
   clearly say that you could not find it.
6. Keep the answer clear and relevant.

Context:
{context}
"""
                        ),
                        (
                            "human",
                            "{question}"
                        )
                    ])

                    # Connect to Ollama
                    llm = OllamaLLM(
                        model=model_name
                    )

                    # Build the RAG chain
                    rag_chain = (
                        prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Generate the answer
                    answer = rag_chain.invoke({
                        "context": context,
                        "question": question
                    })

                # DISPLAY ANSWER DIRECTLY ON WEBPAGE
                st.subheader("Answer")

                st.markdown(answer)

            except Exception as e:
                st.error(
                    f"Error generating answer: {e}"
                )

                st.info(
                    f"Check that Ollama is running and "
                    f"the model '{model_name}' is installed."
                )

else:

    st.info(
        "Upload your documents and click "
        "'Build Knowledge Base' to begin."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Built with Streamlit, LangChain, Hugging Face, "
    "FAISS and Ollama."
)
