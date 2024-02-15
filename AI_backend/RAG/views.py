from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import AI_Logic.main as AI
# Create your views here.
#this is to start the connection and build the AI 
AI_PACK = AI.start_RAG()

def get_ai_response(req):
    resp = AI.get_response(*AI_PACK,3,req.GET.get("question"))
    print(resp)
    return JsonResponse({"response":resp})
