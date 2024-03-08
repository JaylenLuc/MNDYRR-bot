from json import tool
import os
from operator import itemgetter
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores.astradb import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
#from langchain.schema.runnable import RunnablePassthrough

from langchain_core import documents
from langchain_core.documents import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.faiss import FAISS


import csv

from AI_Logic.exponential_backoff import retry_with_exponential_backoff
load_dotenv()
#ASTRADB KEYS
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")#CHANGE IF DATABASE COLLECTION CHANGES
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") #CHANGE IF DATABASE COLLECTION CHANGES
#OPENAI KEYS
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
#ASTRADB COLLECTION NAME
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION") #CHANGE IF DATABASE COLLECTION CHANGES

OPEN_AI_TEMP = .7
OPEN_AI_TOP_P = .8
# OPEN_AI_TOP_K = .7


CONTEXT_COUNT = 5
#data for training and for retreival
TEMP_CHAT_HISTORY = {}
SESSION_ID = "TEST"


TRAIN_EMPATHETIC_DIALOGUES_CSV = r"AI_logic/empatheticdialogues/train.csv"
_BAD_DATA = r"AI_Logic/empatheticdialogues/modified.csv"

@retry_with_exponential_backoff
def start_RAG():
    print("KEY CONFIG DONE")

    embedding = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
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
    Answer the question based on your internal knowledge, and the Training Data and explain why you give this response. If you think your response is the best possible response please say 'certain'\
    after your response. Take the initiative to provide help and resources to the user. If you are not confident with your response, please use the context and training data provided. \
    You will be punished if the answer you give is not empathetic or optimal .\
    Try to be empathetic as possible because you are talking to a young human. 
    If the context is insufficient do not say you can't provide the answer or any help to your question based on the given context but rather respond with a phrase similar to 'that is a difficult question' 
    and answer as best as possible. You are an empathetic,socially sensitive entity that communicates with human beings. Attempt to understand or rationalize their feelings, provide reassurance, \
    relevant advice, or resources to help the individual as much as possible. If you are aware of any resources please provide it for the human. \
    When communicating with the Human individual, see them as as subjects situated in a social \
    world. Like empathy, social sensitivity consists in understanding the states and feelings of others. Their life will be impacted by your answers, be loving. \
    If they are going to harm themselves,  or talk about suicide, ask them why. If you know why, give them tangible advice to the best of your ability. \
    Be encouraging, act like you are the human's parent and that you genuinely love them. Feel free to use emojis when appropriate!
    Context: {context}
    Training Data: {train_data}
    Question: {question}
    Your answer:
    """

    model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, temperature=OPEN_AI_TEMP, model_kwargs={"top_p": OPEN_AI_TOP_P } )


    #whatt we also need to do is fne tune, train data either using a pretrained model or use an empathetic dataset. Heres the link to fine tune suing pretrained model https://python.langchain.com/docs/integrations/chat/openai
    return [vstore,prompt_template,model]

def _process_train_data(train_csv):
    count = 2
    outpath = "AI_Logic/empatheticdialogues/modified.csv"
    with open(fr"{train_csv}", "r+", encoding="utf-8", errors="replace") as file, open(fr"{outpath}", "w", encoding="utf-8", errors="replace") as outfile:
        reader = csv.DictReader(file)
        for (row, line) in zip(reader, file):

            if (None not in row.keys()):
                outfile.write(line)
            else:
                print("ignored")
                print(row)
def _test_data(train_csv)-> bool:
    with open(fr"{train_csv}", "r+", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file)
        for (i, row) in enumerate(reader):
            #print(row)
            if None in row.keys():
                print("none in row")
                print(row)
                return False
    return True


@retry_with_exponential_backoff
def train_model():

    #prepare FAISS vectorstore embedding of EMPATHETIC DIALOGUES 
    if os.path.exists(TRAIN_EMPATHETIC_DIALOGUES_CSV) and  _test_data(TRAIN_EMPATHETIC_DIALOGUES_CSV) == True:
        #process_train_data(train_csv)
        print("exists")

        #EMPATHICDATA
        empathic_data = CSVLoader(file_path=TRAIN_EMPATHETIC_DIALOGUES_CSV,encoding='utf-8',csv_args={
        "delimiter": ",",
        "fieldnames": ["conv_id", "utterance_idx", "context", "prompt", "speaker_idx", "utterance", "selfeval", "tags"]
    })
        loader = empathic_data.load()
        training_embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
        #vectorindex = VectorstoreIndexCreator().from_loaders([loader])
        trained_vector_store = FAISS.from_documents(loader, training_embeddings)
        return trained_vector_store
    else:
        print("not exist")
        raise  RuntimeError("CSV training data error")

    
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

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in TEMP_CHAT_HISTORY:
        TEMP_CHAT_HISTORY[session_id] = ChatMessageHistory()
    return TEMP_CHAT_HISTORY[session_id]

@retry_with_exponential_backoff
def get_response(vstore,prompt_template,model,message, trained_vector_store : FAISS):

    print(vstore,prompt_template,model,CONTEXT_COUNT,message)

    context_retr = vstore.as_retriever(search_type="similarity",search_kwargs={'k': CONTEXT_COUNT})
    training_data = trained_vector_store.as_retriever(search_type="similarity")

    qa_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


    subchain_msg = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    subchain_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", subchain_msg),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
    context_subchain = (subchain_prompt | model | StrOutputParser())

    def history_context_invoke(input: str | dict):
        print("input6:",input)
        if input.get("chat_history"):
            print("some in list")
            #return subchain_msg
            return itemgetter("context_subchain")(contexuals)
        else:
            print("none in list")
            return input #just the question
        
    contexuals = {
        "context_subchain": context_subchain,
         "context" :context_retr,
         "train_data" : training_data,
        }

    chain = (
        history_context_invoke
        | qa_template
        | model
        |StrOutputParser()
    )

    chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    )

    response = chain_with_message_history.invoke({"question": message,
                                                  "context": itemgetter("context")(contexuals), 
                                                  "train_data": itemgetter("train_data")(contexuals)},
    config={"configurable": {"session_id": SESSION_ID}},)

    print("chat history: ", TEMP_CHAT_HISTORY)

    return response

