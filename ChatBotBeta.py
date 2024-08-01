import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    nome_arquivo = '/home/henrique/Chat/Procuradoria Geral - Normas.pdf'
    loaders = [PyPDFLoader(nome_arquivo)]
    # Cria a base vetorizada    
    index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", embedding_ctx_length=256, api_key=api_key),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    ).from_loaders(loaders)
    
    return index

# Carrega o índice apenas uma vez e o reutiliza
index = load_pdf()

# Configuração da cadeia de perguntas e respostas
chain = RetrievalQA.from_chain_type(
    retriever=index.vectorstore.as_retriever(),
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
