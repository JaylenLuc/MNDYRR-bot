import django
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import AI_backend.AI_Logic.main as AI
import AI_backend.RAG.JWT as JWT
from dotenv import load_dotenv
import os
 

load_dotenv()
#ASTRADB KEYS
JWT_SECRET_SALT = os.getenv("JWT_SECRET_SALT")
# Create your views here.
#this is to start the connection and build the AI 


AI_PACK = AI.start_RAG()
FIREBASE_ACTIVE = AI.start_firebase()

#PERFORM ASTARDB WRITES ADHOC-----------------------------

#AI.populate_db_jstest(AI_PACK[0]) #push this first
#AI.populate_db(AI_PACK[0], None) #push this second

#---------------------------------------------------------
TRAIN_VECTOR_STORE = AI.train_model()
print("trained")
INVOCATION_CHAIN_DICT = AI.prepare_chain(*AI_PACK,TRAIN_VECTOR_STORE)
print("invoked")
ENABLED_COOKIES = False
def get_ai_response(req):
    #req.GET.get("question"),
    print(req.method)
    req_body = req.GET.get("JWT")
    geolocator = req.GET.get("geolocation")
    if req.method == 'GET':
        if(req_body != "false" and JWT.authJWTSignature(req_body,JWT_SECRET_SALT) == True):
            print("You're logged in via cookies.")
            ENABLED_COOKIES = True
            INVOCATION_CHAIN_DICT["config"]["configurable"]["user_id"] = JWT.extract_session_id(req_body)
            print(FIREBASE_ACTIVE)
            if (geolocator != ""):
                #decode base64 and decode HMACSHA256
                print("geolocator: ",geolocator)
                INVOCATION_CHAIN_DICT["invoke_arg1"]["geolocation"] = geolocator
        else:
            ENABLED_COOKIES = False
            print("cookies not enabled")
            INVOCATION_CHAIN_DICT["config"]["configurable"]["user_id"] = False
            #return HttpResponse("Please enable your cookies")
            #if it fails you will not get the history populated


    
    #-----SERVER RESPONSE--------------------------------------
    INVOCATION_CHAIN_DICT["invoke_arg1"]["question"] = req.GET.get("question")
    print("req:",req)
    resp = AI.get_response(req_body,ENABLED_COOKIES,**INVOCATION_CHAIN_DICT) #[currentTime, {"AIMessage" : resp , "HumanMessage" : query}]
    print("reps:",resp)
    return JsonResponse({"response":resp})


def set_cookies(req):
    print("setting cookies")
    req_body = req.GET.get("JWT")
    if(req_body == "false"):
        new_JWT = JWT.create_JWT()
        print("JWT: ",new_JWT)
        req_body = JWT.createSignature(new_JWT, JWT_SECRET_SALT)
        return JsonResponse({"response":req_body})
    else:
        #check if they exist in the langfuse database
        auth = JWT.authJWTSignature(req_body,JWT_SECRET_SALT)
        print("auth: ", auth)
        if (auth == True):
            #populate chat history store with a function that access the TEMP_STORE
            jwt_token = JWT.extract_session_id(req_body)
            chat_history = AI.populate_chat_history(jwt_token)
            print("on cookie load: ", jwt_token)

            return JsonResponse({"response": chat_history})
        else:
            return JsonResponse({"response":"FAILED"})
        
    
    
    
