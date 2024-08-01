# Relatório do Projeto de ChatBot RAG
## Sumário
1. Introdução
2. Conceitos utilizados
3. Desenvolvimento
4. Considerações

## Introdução

Este trabalho tem por objetivo criar um chatbot de auxílio para resolução de dúvidas relacionadas ao Vestibular da Unicamp 2025 a partir da publicação da Resolução GR-029/2024, de 10/07/2024. Para elaboração do projeto foi utilizada ferramentas como ChatGPT, Markdown e frameworks como langchain, streamlit, dotenv. Vale citar que a fundação do projeto foi em RAG, se utilizando principalmente de history_aware_retriever.


## Conceitos utilizados

Dentro do projeto foi usado conceitos de RAG, como dito anteriormente, onde obtemos uma resposta à pergunta através da ligação de um generator com retriver. Para utilização deste modelo foi necessário aprender sobre embeddings, vectorstore, splitting e os diferentes métodos de realizar cada uma das funções, além de compreender os limites de cada um dos métodos aplicados. Compreendi também o uso de ApiKEYS para utilizar chats como desenvolvedor.


## Desenvolvimento

O desenvolvimento do projeto pode ser segmentado em 4 marcos representados pelas 4 versões do chatbot incluídas no repositório. Em ordem de progressão elas se apresentam da seguinte maneira:
    * ChatBotAlpha.py
    * ChatBotBeta.py
    * ChatBotOmega.py
    * ChatBot.py

### ChatBotAlpha.py
ChatBotAlpha.py representa a primeira versão funcional do projeto, a reunião do mínimo dos conceitos requisitados para criação de um chatbot funcional. Todavia apresentava certa instabilidade nas respostas quando submetido a testes exaustivos, além da ausência de diversos fatores como prompt próprio, uma ConversationalRetrievalChain envés de RetrievalQA (que apesar de obsoleta cumpria mais acertivamente os objetivos do projeto), pouca elaboração no streamlit e principalmente problemas de embeddiment.

### ChatBotBeta.py
Nesta etapa a compreensão do que eram os conceitos de embedding, vectorstore e splitting começaram a consolidar-se e apesar de permanecer com alguns dos problemas da versão anterior, o desenmvolvimento era notável. Agora os embeddings eram armazenados em cache para evitar que na mesma utilização do programa fosse necessário refazer todo processo para criar os embeddings, além de maior desenvolvimento na parte do streamlit. 
  
Contudo, ainda persistiam problemas de embedding, pois com esta versão, caso fosse necessário reiniciar o programa, todos embeddings seriam perdidos, custando ainda mais para refaze-los nas próxima execução.

### ChatBotOmega.py
Feita as alterações da versão passada, o foco do desnvolvimento voltou-se ao processo de embedding, priorizando a salvamento dos embeddings criados em um arquivo no disco rígido para terem que ser gerados uma única vez. Optei por utilizar da biblioteca do Facebook FAISS para gravar os embeddings em um arquivo .faiss no repositório, permitindo que fosse reutilizado o índice antes criado. Tais alterações foram essenciais para o andamento do projeto já que o próximo passo seria personalizar o prompt da chain e a criação, utilização e valiação de um dataset ouro, um conjunto de perguntas e respostas esperadas e uma função focada em realização de testes.
  

### ChatBot.py
Para prosseguimento do projeto, foi necessária uma reavaliação dos conceitos abordados. Na etapa final identifiquei resquicios de código de versões anteriores que tinham se tornado obsoletos dentro da etapa atual, como a utilização de agentes, assim como o método de criação de uma retrieval-chain, já que RetrievalQA não é atualmente utilizado. Para esta finalidade foi utilizada a combinação de uma question_answer_chain, obtida a partir de uma llm e um prompt customizado e um history_aware_retriever, retriver esse que analisa o histórico de mensagens para geração de perguntas e estimula uma resposta pela outra chain, permitindo um algoritmo mais assertivo e econômico, se reutilizando de respostas anteriores.
  
Conjuntamente tivemos a adição de prompts específicos e o dataset ouro, que permitiu com que eu avaliasse as repostas dada pelo chatbot e atribuisse um score a elas. O método de avaliação entende que caso maior parte da resposta esperada esteja contida na resposta obtida, aquela resposta passa. Neste caso é necessário que a resposta esperada seja sucinta, contendo quase exclusivamente a informação requisitada.

## Considerações
O projeto apresentou e utilizou todos requisitos do processo seletivo e exercitou a minha capacidade de procura, aprendizagem e independência. O resultado julgo satisfatório pela maior parte dos testes terem passado e o chat complementar com informações úteis conforme requisitado em seu prompt.
