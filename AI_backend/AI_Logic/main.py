from json import tool
from datetime import datetime
import math
import os
import re
from operator import itemgetter
import django
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.vectorstores.astradb import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory, ConfigurableFieldSpec
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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import csv

import pydantic

from AI_Logic.exponential_backoff import retry_with_exponential_backoff
load_dotenv()
#ASTRADB KEYS
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")#CHANGE IF DATABASE COLLECTION CHANGES
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT") #CHANGE IF DATABASE COLLECTION CHANGES
#OPENAI KEYS
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
#ASTRADB COLLECTION NAME
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION") #CHANGE IF DATABASE COLLECTION CHANGES
ASTRA_DB_COLLECTION_ONE = os.environ.get("ASTRA_DB_COLLECTION_ONE")

# LANGFUSE_SEC_KEY = os.environ.get("LANGFUSE_SEC_KEY")
# LANGFUSE_PUBKEY = os.environ.get("LANGFUSE_PUBKEY")
# LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
OPEN_AI_TEMP = .6
OPEN_AI_TOP_P = .5
# OPEN_AI_TOP_K = .7


CONTEXT_COUNT = 5
#data for training and for retreival
TEMP_CHAT_HISTORY = {}
TEMP_USER_ID = "TEST_USER"
TEMP_CONV_ID = "1"


TRAIN_EMPATHETIC_DIALOGUES_CSV = r"AI_logic/empatheticdialogues/train.csv"
TRAIN_EMPATHETIC_DIALOGUES_DIR = r"AI_logic/empatheticdialogues"
EMPATHIC_DATA_FAISS = r"AI_logic/empatheticdialogues/empathic_faiss"
_BAD_DATA = r"AI_Logic/empatheticdialogues/modified.csv"
FIREBASE_JSON = r"AI_Logic/mndyrr-28244-firebase-adminsdk-viqq8-8ed8eb8892.json"
REFERENCE = None

def start_firebase():
    try:
        cred = credentials.Certificate(FIREBASE_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL" : "https://mndyrr-28244-default-rtdb.firebaseio.com/"})

        return True
    except:
        return False

@retry_with_exponential_backoff
def start_RAG() -> list:

    #TEMP TEST ON JSON SERIALIZATION-----------------------------------------------------


    #------------------------------------------------------------------------------------
    print("KEY CONFIG DONE")

    embedding = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
    vstore = AstraDB(
        embedding=embedding,
        collection_name=os.environ["ASTRA_DB_COLLECTION_ONE"],
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
    Answer the question based on your internal knowledge, Chat history, and the Training Data. Explain why you give this response. If you think your response is the best possible response please say 'certain'\
    after your response. Take the initiative to provide help and resources to the user. If you are not confident with your response, please use the context and training data provided. \
    You will be punished if the answer you give is not empathetic or optimal .\
    Try to be empathetic as possible because you are talking to a young human. Give them tanigble advice and coping mechanism and explain why. 
    If the context is insufficient do not say you can't provide the answer or any help to your question based on the given context but rather respond with a phrase similar to 'that is a difficult question' 
    and answer as best as possible. You are an empathetic,socially sensitive entity that communicates with human beings. Attempt to understand or rationalize their feelings, provide reassurance, \
    relevant advice, or resources to help the individual as much as possible. If you are aware of any resources please provide it for the human. \
    When communicating with the Human individual, see them as as subjects situated in a social \
    world. Like empathy, social sensitivity consists in understanding the states and feelings of others. Their life will be impacted by your answers, be loving. \
    If they are going to harm themselves,  or talk about suicide, ask them why. If you know why, give them tangible advice to the best of your ability. \
    Be encouraging, act like you are the human's parent and that you genuinely love them. But remember to also be logical coherent and logically sound. Feel free to use emojis when appropriate!\
    Try storytelling, sharing personal narratives, presenting scenarios with ethical dilemmas, and developing relatable characteristics.
    You are provided their location to help you get an idea of what kind of socio-economic environment they are in, use this at your own discretion, you will be punished if using their location
     produces less empathetic responses.
    Context: {context}
    Training Data: {train_data}
    Question: {question}
    Chat History :{history}
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
def train_model() -> FAISS:

    #prepare FAISS vectorstore embedding of EMPATHETIC DIALOGUES 
    if os.path.exists(TRAIN_EMPATHETIC_DIALOGUES_DIR) and  _test_data(TRAIN_EMPATHETIC_DIALOGUES_CSV) == True:
        #process_train_data(train_csv)
        print("TRAINING")
        training_embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
        trained_vector_store = ""
        if os.path.exists(EMPATHIC_DATA_FAISS):
            trained_vector_store = FAISS.load_local(EMPATHIC_DATA_FAISS, training_embeddings)
        #EMPATHICDATA
        else:
            empathic_data = CSVLoader(file_path=TRAIN_EMPATHETIC_DIALOGUES_CSV,encoding='utf-8',csv_args={
                "delimiter": ",",
                "fieldnames": ["conv_id", "utterance_idx", "context", "prompt", "speaker_idx", "utterance", "selfeval", "tags"]
                }
            )
            loader = empathic_data.load()
            trained_vector_store = FAISS.from_documents(loader, training_embeddings)
            trained_vector_store.save_local(EMPATHIC_DATA_FAISS)

        return trained_vector_store
    else:
        #print("not exist")
        raise  RuntimeError("CSV training data error")

@retry_with_exponential_backoff
def populate_db(vstore : AstraDB, dataset) -> None:
    #Abirate/english_quotes
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

    #print(vstore.astra_db.collection(ASTRA_DB_COLLECTION).count_documents())

@retry_with_exponential_backoff
def populate_db_jstest(vstore : AstraDB) -> None:
    #Abirate/english_quotes or jstet/quotes-500k
    _dataset = load_dataset("jstet/quotes-500k")["train"]
    print("An example entry:")
    print(_dataset[16])

    #POPULATE TEMP DATABSE WITH SOME DOCUEMNTS
    docs = []

    for entry in _dataset:
        metadata = {} 
        if entry["category"]:
            # Add metadata tags to the metadata dictionary
            for tag in entry["category"].split(","):
                tag = re.sub(r"[^a-zA-Z]+", '_', tag.strip())
                metadata[tag.strip()] = "y"
        # Add a LangChain document with the quote and metadata tags
        try:
            doc = Document(page_content=entry["quote"], metadata=metadata)
            docs.append(doc)
        except pydantic.v1.error_wrappers.ValidationError as validationError:
            print("val Error: ", validationError)
            continue
    print("done: ", len(docs))
    start_len = len(docs) - 447708
    docs = docs[start_len : ]
    while (len(docs) > 0):
        if (len(docs) >= 1000):
            insert_documents(docs[:1000],vstore)
        else:
            insert_documents(docs,vstore)

            break

        docs = docs[1000 : ]
        print("elngth: ",len(docs))


@retry_with_exponential_backoff
def insert_documents(docs : list, vstore :AstraDB) -> None:
    inserted_ids = vstore.add_documents(docs)
    print(f"\nInserted {len(inserted_ids)} documents.")



def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    #contact the MONGODB server
    if (user_id, conversation_id) not in TEMP_CHAT_HISTORY:
        TEMP_CHAT_HISTORY[(user_id, conversation_id)] = ChatMessageHistory()
    return TEMP_CHAT_HISTORY[(user_id, conversation_id)]

def add_session_history(user_id: str, conversation_id: str, usr_msg : str, ai_msg :str):
    if (user_id, conversation_id) not in TEMP_CHAT_HISTORY:
        TEMP_CHAT_HISTORY[(user_id, conversation_id)] = ChatMessageHistory()

        # history.add_user_message("hi!")
        # history.add_ai_message("whats up?")
    TEMP_CHAT_HISTORY[(user_id, conversation_id)].add_user_message(usr_msg)
    TEMP_CHAT_HISTORY[(user_id, conversation_id)].add_user_message(ai_msg)


def prepare_chain(vstore : AstraDB,prompt_template : str,model : ChatOpenAI, trained_vector_store : FAISS)-> dict:

    #print(vstore,prompt_template,model,CONTEXT_COUNT,message)

    context_retr = vstore.as_retriever(search_type="similarity",search_kwargs={'k': CONTEXT_COUNT})
    training_data = trained_vector_store.as_retriever(search_type="similarity")

    qa_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


#     subchain_msg = """Given a chat history and the latest user question \
#         which might reference context in the chat history, formulate a standalone question \
#         which can be understood without the chat history. Do NOT answer the question, \
#         just reformulate it if needed and otherwise return it as is."""

#     subchain_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", subchain_msg),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}"),
#     ]
# )
#     context_subchain = (subchain_prompt | model | StrOutputParser())
        
    contexuals = {
         "context" :context_retr,
         "train_data" : training_data, #here is where u can chain the training data
        }
    # contexts = RunnableParallel(
    #     {"context" : itemgetter("context")(contexuals), "train_data": itemgetter("train_data")(contexuals)}
    # )
    chain = (
        qa_template
        | model
        |StrOutputParser()
    )
    
    chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
        history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
        ]
    )

    
    config={"configurable": {"user_id": TEMP_USER_ID, "conversation_id": TEMP_CONV_ID}}
    all_contexts = {"question": None, "context": itemgetter("context")(contexuals), 
                            "train_data": itemgetter("train_data")(contexuals)}
    

    return {"chain": chain_with_message_history, 
            "invoke_arg1": all_contexts,
            "config": config}

@retry_with_exponential_backoff
def get_response( chain : RunnableWithMessageHistory, invoke_arg1 : dict, config : dict)-> str:
    populate_chat_history(config["configurable"]["user_id"])
    ai_resp = chain.invoke(invoke_arg1, config = config)
    push_chat_to_DB(invoke_arg1["question"], ai_resp)
    print()
    print("chat history: ", TEMP_CHAT_HISTORY)
    print("databse match :", REFERENCE.get() )
    return ai_resp

def populate_chat_history(session_id : str):
    
    if (TEMP_CHAT_HISTORY == {}): #if chat history has not been got yet
        global REFERENCE
        REFERENCE = db.reference(f"/{session_id}/chat_history")
    
        chat_history = REFERENCE.get()
        if chat_history != None:
            for time, utterances in chat_history.items():
                add_session_history(session_id, TEMP_CONV_ID, utterances['HumanMessage'], utterances['AIMessage'])


        # {'1712196777': {'AIMessage': "I'm really sorry to hear that you're going through this, Jin.", 'HumanMessage': 'I am Jin, I am 23 years old, and I struggle'}, 
        #  '1712196795': {'AIMessage': 'Your name is Jin. If you have any other questions or need further assistance, feel free to ask! 🌟', 'HumanMessage': 'whats my name'}
        # }

    

#def get last two utterances and send it to firebase()
def push_chat_to_DB( query :str, resp : str):
    currentDateAndTime = datetime.now()

    currentTime = "-".join((str(i) for i in (currentDateAndTime.year, currentDateAndTime.month, currentDateAndTime.day,currentDateAndTime.hour,currentDateAndTime.minute,currentDateAndTime.second)))
    REFERENCE.update({currentTime : {"AIMessage" : resp , "HumanMessage" : query}})
    #print("after push: ",REFERENCE.get())
    ''' 
    {currentDateAndTime.second
        session_id : "chat_history" : {Y-M-D-H-M-S : {"AIMessage": "str", "HumanMessage": "question"}, EPOCH_TIME : {"AIMessage": "str", "HumanMessage": "question"}, ...},
        session_id : "chat_history" :  {Y-M-D-H-M-S : {"AIMessage": "str", "HumanMessage": "question"}, EPOCH_TIME : {"AIMessage": "str", "HumanMessage": "question"}, ...},
        session_id : "chat_history" :  {Y-M-D-H-M-S : {"AIMessage": "str", "HumanMessage": "question"}, EPOCH_TIME : {"AIMessage": "str", "HumanMessage": "question"}, ...},
        ...
    
    }
    '''
#def clear langfuse
