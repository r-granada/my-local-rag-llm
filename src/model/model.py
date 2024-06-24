from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os


class Model:
    # _model_id = "meta-llama/Meta-Llama-3-8B"
    _model_id = "nvidia/Llama3-ChatQA-1.5-8B"
    _device = torch.cuda.current_device()

    def __init__(self):
        load_dotenv()
        self._access_token = os.getenv("ACCESS_TOKEN")
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()

    def _load_model(self):
        assert self._tokenizer is not None, "Tokenizer must be initialized first"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=bnb_config,
            device_map={'': self._device}, 
            token=self._access_token,
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        return model

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            token=self._access_token,
        )
 
        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    def answer(self, question: str, context: str = None) -> str:
        # if context:
        #     question = f"Question: {question}.\nContext: {context}."
        messages = [
                {"role": "system", "content": "You are a helpful and intelligent AI assistant who responds to user queries. Use the information provided in the context if available."},
                {"role": "user", "content": question},
                {"role": "context", "content": context},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self._device)
        outputs = self._model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=5, top_p=0.95)
        decoded_result = self._tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        return decoded_result

