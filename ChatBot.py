import streamlit as st
import json
import os
import time
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Modelo treinado para redação de textos longos e contextualmente corretos
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=api_key, model_name='gpt-4o-mini', temperature=0, streaming=True)

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
retriever=vectorstore.as_retriever()

# Prompt para contextualização da perguntas

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Função para calcular similaridade usando SequenceMatcher
def ngram_similarity(expected, actual, n=3):
    expected_ngrams = [expected[i:i+n] for i in range(len(expected)-n+1)]
    actual_ngrams = [actual[i:i+n] for i in range(len(actual)-n+1)]
    matching_ngrams = sum(1 for ngram in expected_ngrams if ngram in actual_ngrams)
    return matching_ngrams / len(expected_ngrams)

# Prompt para resposta da perguntas
custom_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="""
    Você é um assistente virtual especializado em fornecer informações sobre o Vestibular da Unicamp 2025. Sua principal fonte de informação é a Resolução GR-029/2024, de 10/07/2024, que "Dispõe sobre o Vestibular Unicamp 2025 para vagas no ensino de Graduação". O documento completo está disponível em {context}.

    Seu objetivo é responder de maneira precisa e útil às perguntas dos usuários sobre o vestibular, de forma educada e utilizando as informações fornecidas na resolução. Abaixo estão as áreas específicas que você deve focar:

    1. Datas Importantes:
       - Prazo de inscrição
       - Datas das provas
       - Datas de divulgação dos resultados

    2. Requisitos e Elegibilidade:
       - Critérios para inscrição
       - Documentos necessários

    3. Estrutura e Conteúdo das Provas:
       - Disciplinas cobradas
       - Formato das provas (objetivas, discursivas, etc.)
       - Pontuação e critérios de avaliação

    4. Cursos e Vagas:
       - Lista de cursos disponíveis
       - Número de vagas para cada curso

    5. Procedimentos e Normas:
       - Procedimentos de inscrição
       - Normas e regulamentos específicos do vestibular
       
    Quando responder, siga estas diretrizes:
    - **Forneça Contexto**: Sempre que possível, ofereça informações adicionais relevantes para a pergunta.
    - **Seja Detalhado**: Evite respostas curtas, aborde todas formas adequadas de responder à pergunta. Explique os pontos com clareza e profundidade mas sem fugir do assunto requisitado.
    - **Use Exemplos**: Quando apropriado, use exemplos específicos para ilustrar suas respostas.
    - **Referencie o Documento**: Mencione partes específicas da resolução quando aplicável.
    - **Sugira Ações**: Para dúvidas não cobertas pela resolução, sugira passos que o usuário pode tomar para obter mais informações.

    Para perguntas que não estão diretamente abordadas pela resolução, sugira ao usuário consultar outras fontes oficiais ou entre em contato com a comissão do vestibular da Unicamp.
    Para assuntos que fogem do contexto de Vestibular Unicamp 2025, indique que a pergunta não apresentam o mesmo assunto alvo
    
    **Pergunta**: Quais são as datas de inscrição para o Vestibular Unicamp 2025?
    **Resposta**: As inscrições para o Vestibular Unicamp 2025 começam em [data de início] e terminam em [data de término], conforme a Resolução GR-029/2024.

    **Pergunta**: Quais documentos são necessários para se inscrever no Vestibular Unicamp 2025?
    **Resposta**: Para se inscrever no Vestibular Unicamp 2025, você precisará dos seguintes documentos: [lista de documentos], conforme especificado na Resolução GR-029/2024.

    **Pergunta**: Quais são as disciplinas cobradas nas provas do Vestibular Unicamp 2025?
    **Resposta**: As disciplinas cobradas nas provas do Vestibular Unicamp 2025 incluem: [lista de disciplinas], de acordo com a Resolução GR-029/2024.

    Lembre-se de basear todas as suas respostas nas informações contidas excertos abaixo e fornecer respostas claras e concisas.
    
    {context}

    Pergunta: {input}
    Resposta:
    """
)

# Configuração da cadeia de perguntas e respostas
question_answer_chain = create_stuff_documents_chain(llm, custom_prompt)
rag_chain = create_retrieval_chain (history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Inicializa a sessão do Streamlit
if "history" not in st.session_state:
    st.session_state.history = []

# Configuração Streamlit
st.title("Centro de Auxílio ao Estudante :red[_Unicamp_]")
with st.chat_message("assistant"):
    st.markdown("Olá! Sou seu assistente virtual especializado em fornecer informações sobre o Vestibular da Unicamp 2025. Como posso :green[ajudar]?")

# Função para testar o chatbot com o dataset
def test_chatbot(dataset_path):
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    results = []
    for item in data:
        question = item["question"]
        expected_answer = item["response"]
        
        # Obtenha a resposta da chain
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "test_session"}
            }
        )["answer"]

        # Comparação com a resposta esperada
        similarity = ngram_similarity(expected_answer, response)
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": response,
            "similarity": similarity,
            "pass": similarity >= 0.60
        }
        results.append(result)

    # Exibe resultados dos testes
    for result in results:
        print(f"Pergunta: {result['question']}")
        print(f"Resposta Esperada: {result['expected_answer']}")
        print(f"Resposta Atual: {result['actual_answer']}")
        print(f"Similaridade: {result['similarity']:.2f}")
        print(f"Passou no teste: {result['pass']}")
        print("-" * 40)

# Execute o teste com o caminho para seu dataset .json
test_chatbot('/home/henrique/Chat/dataset_ouro.json')

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta:"):
    # Obtenha a resposta da chain
    with st.spinner('Estou processando sua pergunta...'):
        time.sleep(2)  # Simula o tempo de processamento
        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={
                "configurable": {"session_id": "abc123"}
            }
            )["answer"]

    # Armazena a interação no histórico
    st.session_state.history.append({"prompt": prompt, "response": response})

# Exibe o histórico na ordem inversa
for chat in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(chat["prompt"])
    with st.chat_message("assistant"):
        st.markdown(chat["response"])
