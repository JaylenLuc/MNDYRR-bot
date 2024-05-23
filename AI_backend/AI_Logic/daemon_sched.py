def send_astra():
    import os
    from dotenv import load_dotenv
    from langchain_community.vectorstores.astradb import AstraDB
    from langchain_openai import OpenAIEmbeddings
    import time
    load_dotenv()
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")#CHANGE IF DATABASE COLLECTION CHANGES
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") #CHANGE IF DATABASE COLLECTION CHANGES
    #OPENAI KEYS
    OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
    #ASTRADB COLLECTION NAME
    ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION") #CHANGE IF DATABASE COLLECTION CHANGES
    ASTRA_DB_COLLECTION_ONE = os.getenv("ASTRA_DB_COLLECTION_ONE")
    embedding = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
    while (True):
       
        vstore = AstraDB(
            embedding=embedding,
            collection_name=ASTRA_DB_COLLECTION_ONE,
            token=ASTRA_DB_APPLICATION_TOKEN,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
        )
        res = vstore.similarity_search("Everything's ok", k=1)
        print("stub read res: ",res)
        print("send ", time.time()) #push for next commit
        time.sleep(144000.0)
    return
if __name__ == "__main__":
    send_astra()