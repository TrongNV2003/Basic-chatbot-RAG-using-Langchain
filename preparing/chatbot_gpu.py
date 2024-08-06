import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# Recommended to execute on google colab

class Chatbot:
    def __init__(self):
        MODEL_QUANTIZED = "models/Llama-1.1B-AWQ-4bit"
        self.model = AutoAWQForCausalLM.from_pretrained(MODEL_QUANTIZED, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_QUANTIZED, use_fast=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def runbot(self, question: str):
        formatted_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"

        tokens = self.tokenizer(
            formatted_prompt,
            return_tensors='pt'
        ).input_ids.cuda()

        generation_output = self.model.generate(
            tokens,
            max_new_tokens = 512,
            temperature = 0.6,
            top_p = 0.95,
            top_k = 50,
            repetition_penalty=0.95
        )

        output = self.tokenizer.decode(generation_output[0], skip_special_token=True)
        return output        
    
if __name__ == "__main__":
    chatbot = Chatbot()
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        question = input("User: ")
        if question.lower() == "exit":
            break
        response = chatbot.runbot(question)
        print(response)   