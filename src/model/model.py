from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

class Model:

    _model_id = "meta-llama/Meta-Llama-3-8B"

    def __init__(self):
        load_dotenv()
        self._access_token = os.getenv("ACCESS_TOKEN")

        print("Loading model")
        self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype = "auto",
                token = self._access_token,
                load_in_4bit=True,
                device_map="auto")
        print("Loading tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id,
                torch_dtype = "auto",
                token = self._access_token,
                load_in_4bit=True,
                device_map="auto")
        print("Finished setting up Model")



    def answer(self, question: str) -> str:
        inputs = self._tokenizer(question, return_tensors="pt").input_ids
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

