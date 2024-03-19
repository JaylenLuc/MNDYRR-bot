from random import randbytes
import base64
import hmac
import hashlib
import json
import re
def create_JWT() -> dict:
    #geographic area
    return {
        'header' : {
            'alg' : 'HS246',
            'typ' : 'JWT'
                    
        },
        'payload' : {
            'session_id' : randbytes(16).hex() 
        }
    }

def createSignature(jwt : dict, secret_slat : bytes) -> str :
    header = jwt["header"]
    payload = jwt['payload']

    encoded_header = str(base64.b64encode(str(header).strip().encode("utf-8")), 'utf-8')
    print(encoded_header)
    encoded_payload = str(base64.b64encode(str(payload).strip().encode("utf-8")), 'utf-8')
    print(encoded_payload)

    conact_jwt = encoded_header + "." + encoded_payload


    #REFACTOR-----------ABSTRACT INTO FUNCTIOn
    sha_hex = hmac.new(bytes(secret_slat, 'utf-8'), conact_jwt.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

    print(sha_hex)

    signature = str(base64.b64encode(sha_hex.encode("utf-8")), 'utf-8')
    print(signature)
    return encoded_header + "." + encoded_payload + "."+signature

def authJWTSignature(jwt :str, secret_slat : str)-> bool:
    
    jwt_array = jwt.split('.')

    header = jwt_array[0]
    body = jwt_array[1]
    signature = jwt_array[2]
    decoded_header = base64.b64decode(header).decode("utf-8")
    print(decoded_header)
    decoded_header = re.sub("'", '"', decoded_header)

    header_obj = json.loads(decoded_header)
    print("header obj = ",header_obj)
    conact_jwt = header + '.' + body
    print(conact_jwt)
    if (header_obj["alg"] != "HS246" or header_obj['typ'] != "JWT"):
        return False
    
    #REFACTOR-----------ABSTRACT INTO FUNCTIOn
    sha_hex = hmac.new(bytes(secret_slat, 'utf-8'), conact_jwt.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
    new_signature = str(base64.b64encode(sha_hex.encode("utf-8")), 'utf-8')
    print(new_signature, " ", signature)
    if signature != new_signature:
        return False

    return True

def extract_session_id(jwt : str) -> str:
    jwt_payload = jwt.split('.')[1]
    decoded_payload = base64.b64decode(jwt_payload).decode("utf-8")
    decoded_payload = re.sub("'", '"', decoded_payload)
    decoded_payload = json.loads(decoded_payload)
    print("decoded: ",decoded_payload)

    return decoded_payload["session_id"]







