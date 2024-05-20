# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to generate a message with Anthropic Claude (on demand).
"""
import logging

import boto3
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

bedrock = boto3.client(service_name='bedrock-runtime')
# loaders = [PyPDFLoader("ciencia_comida_aviones.pdf"),]
#
# docs = []
#
# for loader in loaders:
#     docs.extend(loader.load())
st.text("Welcome to the RAG demo with Anthropic Claude. Please upload some PDFs to get started.")
files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if len(files) > 0:
    docs = []
    for doc in files:
        reader = PdfReader(doc)
        i = 1
        for page in reader.pages:
            docs.append(Document(page_content=page.extract_text(), metadata={'page': i}))
            i += 1
    r_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                chunk_overlap=100,
                                                separators=["\n\n", "\n"])

    docs_splitted = r_splitter.split_documents(documents=docs)
    vector_store = Chroma.from_documents(documents=docs_splitted,
                                         embedding=BedrockEmbeddings(),
                                         persist_directory='vector_store/chroma/')
    st.text("Loaded the following documents:")
    for file in files:
        st.text(file.name)

    QUERY_PROMPT_TEMPLATE = """\
    H:
    Answer the question based on the provided context. Do not create false information.
    {context}
    Question: {question}
    A:
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=BedrockChat(model_id='anthropic.claude-3-haiku-20240307-v1:0', client=bedrock),
        retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)}
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    st.title("Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = qa_chain(prompt).get('result')
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.text("Please upload some PDFs.")