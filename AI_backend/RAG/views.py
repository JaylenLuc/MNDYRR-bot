from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import AI_Logic.main as AI
# Create your views here.
#this is to start the connection and build the AI 
AI_PACK = AI.start_RAG()
AI.populate_db(AI_PACK[0], None)
TRAIN_VECTOR_STORE = AI.train_model()


def get_ai_response(req):
    resp = AI.get_response(*AI_PACK,req.GET.get("question"),TRAIN_VECTOR_STORE)
    print(resp)
    return JsonResponse({"response":resp})
