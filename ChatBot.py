import streamlit as st
from langchain import hub, chains
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Modelo treinado para redação de textos longos er corentestanto à norma quanto ao contexto
llm = OpenAI(api_key='sk-svcacct-O4NsuqTsAq59ShOSv7pCT3BlbkFJptXX5UOGvCsMHQGegbzD',temperature=0, streaming=True)

# Ferramentas a serem utilizadas
tools = load_tools(["ddg-search"])

# Prompt do agente
prompt = hub.pull("hwchase17/react")

# Criação do agente
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#
knowledge_base = hub.pull()

# Repartição do texto para facilitar indexação
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(knowledge_base)

# Representação vetorial do texto para facilitar busca de elementos
embeddings = OpenAIEmbeddings(api_key='sk-svcacct-O4NsuqTsAq59ShOSv7pCT3BlbkFJptXX5UOGvCsMHQGegbzD')
docsearch = FAISS.from_texts(texts, embeddings)

# Cadeia de recuperação (conecta retriver e gerador para otimização das respostas)
retrieval_chain = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(),
    llm=llm,
    chain_type="qa"
)

st.title("Centro de Auxílio ao Estudante :red[_Unicamp_] :")

if prompt := st.chat_input():                              
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
       # st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}
        )
        st.write(response["output"])