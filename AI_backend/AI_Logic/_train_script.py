TRAIN_PATH = r"AI_logic/empatheticdialogues/train.csv"
TEST_PATH = r"AI_logic/empatheticdialogues/test.csv"
from openai import OpenAI
from langfuse import Langfuse
client = OpenAI()
def _train_openai(training_path : str , test_path : str) -> str:
    client.fine_tuning.jobs.list(limit=10)
    return ""



if __name__ == "__main__":
    _train_openai(TRAIN_PATH,TEST_PATH)
