# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import shutil

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)


bedrock = boto3.client(service_name='bedrock-runtime')
# loaders = [PyPDFLoader("ciencia_comida_aviones.pdf"),]
#
# docs = []
#
# for loader in loaders:
#     docs.extend(loader.load())
st.text("Welcome to the Payments AI Demo.")
st.text(f"The current model is: {MODEL_ID}")
files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if len(files) > 0:
    try:
        shutil.rmtree("vector_store")
    except:
        pass
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
    vector_store = None
    vector_store = Chroma.from_documents(documents=docs_splitted,
                                         embedding=BedrockEmbeddings())
    st.text("Loaded the following documents:")
    for file in files:
        st.text(f'\t- \"{file.name}"')

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=BedrockChat(model_id=MODEL_ID, client=bedrock),
    #     retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
    #     memory=st.session_state.memory,
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)}

    # )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=BedrockChat(model_id=MODEL_ID, client=bedrock),
        memory=st.session_state.memory,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        condense_question_prompt=PromptTemplate(input_variables=['chat_history', 'question'], template='Your task is to help out the user with understanding a document.\nHistory: {chat_history}\nQuestion: {question}')
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    st.title("Your AI Companion")

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

        response = qa_chain(prompt).get('answer')
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # st.info(st.session_state.memory.abuffer_as_str)

else:
    st.text("Please upload some PDFs.")