import streamlit as st
import numpy as np
import pickle
import os
import time
import faiss
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Modelo treinado para redação de textos longos e contextualmente corretos
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=api_key, model_name='gpt-4o-mini', temperature=0, streaming=True)

# Ferramentas a serem utilizadas
tools = load_tools(["ddg-search"])

# Prompt do agente
prompt = hub.pull("hwchase17/react")

# Função para carregar e criar índice vetorial a partir do PDF
@st.cache_resource
def load_pdf():
    index_file_path = '/home/henrique/Chat/vectorstore_index.faiss'
    index_metadata_path = '/home/henrique/Chat/vectorstore_metadata.pkl'
    
    if os.path.exists(index_file_path) and os.path.exists(index_metadata_path):
        # Carrega o índice FAISS e cria o vetorstore FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = FAISS.load_local(index_file_path, embeddings, "index")
        
    else:
        nome_arquivo = '/home/henrique/Chat/Procuradoria Geral - Normas.pdf'
        loaders = [PyPDFLoader(nome_arquivo)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        # Carrega e divide o texto do PDF
        for loader in loaders:
            docs = loader.load()
            documents = text_splitter.split_documents(docs)
        
        # Cria embeddings e a base vetorizada FAISS 
        embedding = OpenAIEmbeddings(model="text-embedding-3-large", embedding_ctx_length=256, api_key=api_key)
        
        # Cria o vetorstore FAISS
        vectorstore = FAISS.from_documents(documents ,embedding)
        vectorstore.save_local(index_file_path, "index")
        
    return vectorstore

# Carrega o índice apenas uma vez e o reutiliza
vectorstore = load_pdf()

# Configuração da cadeia de perguntas e respostas
chain = RetrievalQA.from_chain_type(
    retriever=vectorstore.as_retriever(),
    chain_type='stuff',
    llm=llm,
    input_key='question'
)

# Criação do agente
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Inicializa a sessão do Streamlit
if "history" not in st.session_state:
    st.session_state.history = []

# Configuração Streamlit
st.title("Centro de Auxílio ao Estudante :red[_Unicamp_]")
with st.chat_message("assistant"):
    st.markdown("Olá! Sou seu assistente virtual especializado em fornecer informações sobre o vestibular da Unicamp 2025. Como posso :green[ajudar]?")

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta:"):
    # Obtenha a resposta da chain
    with st.spinner('Estou processando sua pergunta...'):
        time.sleep(2)  # Simula o tempo de processamento
        response = chain.run(question=prompt)

    # Armazena a interação no histórico
    st.session_state.history.append({"prompt": prompt, "response": response})

# Exibe o histórico na ordem inversa
for chat in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(chat["prompt"])
    with st.chat_message("assistant"):
        st.markdown(chat["response"])
