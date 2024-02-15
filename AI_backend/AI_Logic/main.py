import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores.astradb import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core import documents
from langchain_core.documents import Document

def start_RAG():
    load_dotenv()
    #ASTRADB KEYS
    ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")#CHANGE IF DATABASE COLLECTION CHANGES
    ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") #CHANGE IF DATABASE COLLECTION CHANGES
    #OPENAI KEYS
    OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
    #ASTRADB COLLECTION NAME
    ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION") #CHANGE IF DATABASE COLLECTION CHANGES
    print("KEY CONFIG DONE")

    embedding = OpenAIEmbeddings()
    vstore = AstraDB(
        embedding=embedding,
        collection_name=os.environ["ASTRA_DB_COLLECTION"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )
    print("DB STORE DONE")

    philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
    print("An example entry:")
    print(philo_dataset[16])

    #POPULATE TEMP DATABSE WITH SOME DOCUEMNTS
    docs = []

    for entry in philo_dataset:
        metadata = {"author": entry["author"]} 
        if entry["tags"]:
            # Add metadata tags to the metadata dictionary
            for tag in entry["tags"].split(";"):
                metadata[tag] = "y"
        # Add a LangChain document with the quote and metadata tags
        doc = Document(page_content=entry["quote"], metadata=metadata)
        docs.append(doc)

    inserted_ids = vstore.add_documents(docs)
    print(f"\nInserted {len(inserted_ids)} documents.")
    #print(vstore.astra_db.collection(ASTRA_DB_COLLECTION).find_one())
    print(vstore.astra_db.collection(ASTRA_DB_COLLECTION).count_documents())
    prompt_template = """
    Answer the question using the supplied context at your own discretion. If you don't know the answer, give your best answer.
    Context: {context}
    Question: {question}
    Your answer:
    """

    model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY)

    return [vstore,prompt_template,model]

def get_response(vstore,prompt_template,model,context_int,message):
    print(vstore,prompt_template,model,context_int,message)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    context_retr = vstore.as_retriever(search_kwargs={'k': context_int})
    chain = (
        {"context": context_retr, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    print(message)
    response = chain.invoke(message)
    return response


# r = start_RAG()
# print(get_response(*r,3,"How does one reduce their own suffering"))
