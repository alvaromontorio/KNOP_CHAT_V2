from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()


def get_model(model_name, temperature, max_tokens):
    print(f"Parámetros de modelo {model_name, temperature, max_tokens}")
    llm = {
        "llama3-70b-8192": ChatGroq(temperature=temperature,model_name="llama3-70b-8192", max_tokens=max_tokens),
        "llama3-8b-8192": ChatGroq(temperature=temperature,model_name="llama3-8b-8192", max_tokens=max_tokens),
        "mixtral-8x7b-32768": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
        "gemma-7b-it": ChatGroq(temperature=temperature,model_name="mixtral-8x7b-32768", max_tokens=max_tokens),
    }
    return llm[model_name]

index_path = "./index/index_recetas"

embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(
    persist_directory=index_path,
    embedding_function=embeddings
    )
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
        }
    )

# First we need a prompt that we can pass into an LLM to generate this search query

prompt_query = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            """Dado el contenido anterior, genera una consulta de búsqueda para obtener información relevante para la conversación.\
               La consulta debe utilizar palabras clave para que se entienda la esencia del mensaje.
            """,
        ),
    ]
)



prompt_main = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Responde a la pregunta del usuario utilizando ÚNICAMENTE el contexto que tienes a continuación:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)

def get_rag_chain(model_name, temperature, max_tokens):
    retriever_chain = create_history_aware_retriever(get_model(model_name, temperature, max_tokens), retriever, prompt_query)

    document_chain = create_stuff_documents_chain(get_model(model_name, temperature, max_tokens), prompt_main)

    rag_chain = create_retrieval_chain(retriever_chain, document_chain)

    return rag_chain


def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question,messages, model_name="llama3-70b-8192", temperature=0, max_tokens=8192):
    chain = get_rag_chain(model_name, temperature, max_tokens)
    history = create_history(messages)
    aux = {}
    #response = chain.invoke({"question": question,"top_k":3,"messages":history.messages})
    response = ""
    for chunk in chain.stream({"input": question,"chat_history":history.messages}):
        
        if "answer" in chunk.keys():
            response += chunk["answer"]
            yield chunk["answer"]

    #figure = generate_chart()
    #aux["figure"] = figure
    history.add_user_message(question)
    history.add_ai_message(response)

    invoke_chain.response = response
    invoke_chain.history = history
    invoke_chain.aux = aux