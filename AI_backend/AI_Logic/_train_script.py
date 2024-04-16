import openai
import os 
from dotenv import load_dotenv
import csv
import json
import copy


load_dotenv()
TRAIN_PATH = r"empatheticdialogues/train.csv"
TEST_PATH = r"empatheticdialogues/test.csv"
VALID_PATH = r"empatheticdialogues/valid.csv"
TRAIN_EMPATHETIC_DIALOGUES_DIR = r"empatheticdialogues"
FORMATTED_TRAIN_DATA = r"empatheticdialogues/formated_train_data.jsonl"
FORMATTED_TEST_DATA = r"empatheticdialogues/formated_test_data.jsonl"
FORMATTED_VALID_DATA = r"empatheticdialogues/formated_valid_data.jsonl"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key= OPENAI_API_KEY)
def _train_openai() -> str:
    print(client.files.create(
        file=open(FORMATTED_TRAIN_DATA, "rb"),
        purpose="fine-tune"
    ))
    # print(client.files.create(
    #     file=open(FORMATTED_TEST_DATA, "rb"),
    #     purpose="fine-tune"
    # ))
    print(client.files.create(
        file=open(FORMATTED_VALID_DATA, "rb"),
        purpose="fine-tune"
    ))
    #client.fine_tuning.jobs.list(limit=10)

    '''
    FileObject(id='file-ogLA8W3uUbfEHsDveOZzKwkP', bytes=17122663, created_at=1713226472, filename='formated_train_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
    FileObject(id='file-tfa2DDvl7XAEjGyiI5kXlgze', bytes=2533669, created_at=1713226474, filename='formated_valid_data.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)

    FineTuningJob(id='ftjob-Rs6k6my3rd2TSHlt5rlfrvED', created_at=1713226585, error=Error(code=None, message=None, param=None, error=None), 
    fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs='auto', batch_size='auto', learning_rate_multiplier='auto'), 
    model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-HstPDf12TxQSRwVSYEM1AOlo', result_files=[], status='validating_files',
      trained_tokens=None, training_file='file-ogLA8W3uUbfEHsDveOZzKwkP', validation_file='file-tfa2DDvl7XAEjGyiI5kXlgze', user_provided_suffix=None, 
      seed=653073683, integrations=[])
    '''
def _create_job() :
    print(client.fine_tuning.jobs.create(
    training_file='file-ogLA8W3uUbfEHsDveOZzKwkP', 
    validation_file='file-tfa2DDvl7XAEjGyiI5kXlgze',
    model="gpt-3.5-turbo"
    ))
def _print_check_jobs() :
    print(client.fine_tuning.jobs.list(limit=10))
    #client.fine_tuning.jobs.retrieve("ftjob-abc123")

def _format_data(_path : str , format_path:str) -> None:
    current_write = None
    F1 =  {"messages": [ {"role": "system", "content": "Mendy is an Empathetic, socially sensitive entity that communicates with human beings and attempts to understand or rationalize their feelings, provide reassurance, relevant advice, or resources to help the individual as much as possible. "}]}
    H1 =    {"role": "user", "content": "I am feeling {sentiment}. {current_line}"}
    HELP = {"role": "assistant", "content": "{current_line}"}
    openai_format1 =  {"messages": [ {"role": "system", "content": "Mendy is an Empathetic, socially sensitive entity that communicates with human beings and attempts to understand or rationalize their feelings, provide reassurance, relevant advice, or resources to help the individual as much as possible. "}]}
    openai_format_human =    {"role": "Human who is feeling {sentiment}", "content": "{current_line}"}
    openai_format_helper = {"role": "Helper", "content": "{current_line}"}

    
    if os.path.exists(TRAIN_EMPATHETIC_DIALOGUES_DIR) and os.path.exists(FORMATTED_TRAIN_DATA):
        print("exists")
        with open(_path, 'r') as read_file, open(format_path, 'w') as write_file:
            csv_reader = csv.DictReader(read_file, delimiter=',')
            current_speaker = 'human'
            idx = 1
            for row in csv_reader:
                #conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags
                #print(idx, row)
                
                if row['utterance_idx'] == '1':
                

                    if current_write != None and idx > 2: 
                        #print(idx)
                        json_str = json.dumps(current_write)
                        #print("current_json string ",json_str)
                        write_file.write(json_str)
                        write_file.write('\n')
                        current_write = None
                        # print("should be None: ", current_write)
                        # print("should be not long : ",openai_format1)
                    idx = 1
                    openai_format1 = copy.deepcopy(F1)
                    openai_format_human = copy.deepcopy(H1)
                    openai_format_helper = copy.deepcopy(HELP)
                    current_write = openai_format1
                    openai_format_human['content'] = openai_format_human['content'].format(sentiment = row['context'], current_line= row['prompt'] + " " + row['utterance'])
                    #print("first write: ",openai_format_human)
                    current_write['messages'].append( copy.deepcopy(openai_format_human) )
                    current_speaker = 'chatbot'
                    openai_format_human = copy.deepcopy(H1)
                elif (current_speaker == 'chatbot'):
                    openai_format_helper['content'] = openai_format_helper['content'].format(current_line = row['utterance'])
                    current_write['messages'].append(copy.deepcopy(openai_format_helper)) 
                    current_speaker = 'human'
                    openai_format_helper = copy.deepcopy(HELP)
                else: #human but not the first utteranc
                    openai_format_human['content'] = openai_format_human['content'].format(sentiment = row['context'], current_line = row['utterance'])
                    current_write['messages'].append(copy.deepcopy(openai_format_human)) 
                    current_speaker = 'chatbot'
                    openai_format_human = copy.deepcopy(H1)
                idx += 1
                # if (idx == 4): 
                #     print("bruh")
                #     break


    
if __name__ == "__main__":
    #_format_data(VALID_PATH, FORMATTED_VALID_DATA)
    #_format_data(TRAIN_PATH, FORMATTED_TRAIN_DATA)
    print("PREPROCESSING DONE")
    #_train_openai()
    ##_print_check_jobs()

    #_format_data("")
    #_create_job()
    #_print_check_jobs()
    print(client.fine_tuning.jobs.retrieve('ftjob-Rs6k6my3rd2TSHlt5rlfrvED'))
    print()
    print(client.fine_tuning.jobs.retrieve('ftjob-mbP8MmFexJFgb0TrJKtFY77r'))


    ''' 
    fine_tuned_model='ft:gpt-3.5-turbo-0125:personal::9ESZoNRm'

    FineTuningJob(id='ftjob-Rs6k6my3rd2TSHlt5rlfrvED', created_at=1713226585, error=Error(code=None, message=None, param=None, error=None), 
    fine_tuned_model='ft:gpt-3.5-turbo-0125:personal::9ESZoNRm', finished_at=1713233468, hyperparameters=Hyperparameters(n_epochs=1, 
    batch_size=13, learning_rate_multiplier=8), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-HstPDf12TxQSRwVSYEM1AOlo', 
    result_files=['file-D75cuF7QfuKFM1Tpq8IHlzhy'], status='succeeded', trained_tokens=3281924, training_file='file-ogLA8W3uUbfEHsDveOZzKwkP', 
    validation_file='file-tfa2DDvl7XAEjGyiI5kXlgze', user_provided_suffix=None, seed=653073683, integrations=[])
    '''
