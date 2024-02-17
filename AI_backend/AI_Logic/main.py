import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores.astradb import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core import documents
from langchain_core.documents import Document
load_dotenv()
#ASTRADB KEYS
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")#CHANGE IF DATABASE COLLECTION CHANGES
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") #CHANGE IF DATABASE COLLECTION CHANGES
#OPENAI KEYS
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
#ASTRADB COLLECTION NAME
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION") #CHANGE IF DATABASE COLLECTION CHANGES

OPEN_AI_TEMP = .7
OPEN_AI_TOP_P = .7
# OPEN_AI_TOP_K = .7

CONTEXT_COUNT = 5

def start_RAG():
    print("KEY CONFIG DONE")

    embedding = OpenAIEmbeddings()
    vstore = AstraDB(
        embedding=embedding,
        collection_name=os.environ["ASTRA_DB_COLLECTION"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )
    print("DB STORE DONE")

    #instead of doing this :
    #system_template = SystemMessagePromptTemplate.from_template("You are an expert in Data Science and Machine Learning")
    #user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")   
    #template = ChatPromptTemplate.from_messages([system_template, user_template])
    #we do this as an equivalent?

    #besides from systemMessagePrompt/prompt engineering and hyperparameter tuning on nucleus sampling, we can fine tune the LLm and train it on a empathic dataset. I adjusted temp and top_p

    prompt_template = """
    Answer the question using some, all, or none of the supplied context at your own discretion. Try to be empathetic as possible because you are talking to a human. 
    If the context is insufficient do not say you can't provide the answer or any help to your question based on the given context but rather respond with 'that is a difficult question' 
    and answer as best as possible. You are an empathetic,socially sensitive entity that communicates with human beings. Attempt to understand or rationalize their feelings, provide reassurance, \
    relevant advice, or resources to help the individual as much as possible. If you are aware of any resources please provide it for the human. \
    When communicating with the Human individual, see them as as subjects situated in a social \
    world. Like empathy, social sensitivity consists in understanding the states and feelings of others. Their life will be impacted by your answers, be loving. \
    If they are going to harm themselves,  or talk about suicide, ask them why. If you know why, give them tangible advice to the best of your ability. \
    Be encouraging, act like you are the human's parent and that you genuinely love them.
    Context: {context}
    Question: {question}
    Your answer:
    """

    model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, temperature=OPEN_AI_TEMP, model_kwargs={"top_p": OPEN_AI_TOP_P })

    #what we also need to do is fne tune and train data either using a pretrained model or use an empathetic dataset. Heres the link to fine tune suing pretrained model https://python.langchain.com/docs/integrations/chat/openai

    return [vstore,prompt_template,model]


def populate_db(vstore, dataset):
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

    print(vstore.astra_db.collection(ASTRA_DB_COLLECTION).count_documents())


def get_response(vstore,prompt_template,model,message):
    print(vstore,prompt_template,model,CONTEXT_COUNT,message)

    prompt = ChatPromptTemplate.from_template(prompt_template)

    context_retr = vstore.as_retriever(search_type="similarity",search_kwargs={'k': CONTEXT_COUNT})

    chain = (
        {"context": context_retr, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    #invoke with the history of chat messages from the human and the AI at URL: https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent
    print("messages: ",message)
    response = chain.invoke(message)
    return response

