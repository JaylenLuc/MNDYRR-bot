from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import AI_Logic.main as AI
# Create your views here.
#this is to start the connection and build the AI 
AI_PACK = AI.start_RAG()
#AI.populate_db(AI_PACK[0], None)
TRAIN_VECTOR_STORE = AI.train_model()
INVOCATION_CHAIN_DICT = AI.prepare_chain(*AI_PACK,TRAIN_VECTOR_STORE)

def get_ai_response(req):
    #req.GET.get("question"),
    INVOCATION_CHAIN_DICT["invoke_arg1"]["question"] = req.GET.get("question")
    resp = AI.get_response(**INVOCATION_CHAIN_DICT)
    print(resp)
    return JsonResponse({"response":resp})
