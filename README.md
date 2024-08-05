# Basic chatbot using RAG and Langchain

This is a basic chatbot to answer the question with provided knowledge in pdf file. The model using llama-7b quantized from huggingface.

## Usage
First, you need to install requirement pakage:

```bash
pip install -r requirement.txt
```

### Download model
you need to get model llama-7b and save it in Folder "models" before run chatbot. Install from [HuggingFace Transformers](https://huggingface.co/vilm/vinallama-7b-chat-GGUF/tree/main). 

### Vector DB
Then, you need to vectorize the data provided from your need. There are 2 ways: vectorize a context or vectorize a large pdf file. To vectorize, run:

```bash
python create_db.py
```

### Run chatbot
I using vinallama-7b quantized model to run local on my computer. Model will retrieve to the vector DB and answer the question regarded to provided knowledge. To ask chat bot, please provide your question in "runbot.py" and run:

```bash
!python runbot.py
```

If you want to run on CPU, change in "runbot.py":
```python
from preparing.chatbot_cpu import Chatbot
```

Else you want to run on GPU:
```python
from preparing.chatbot_gpu import Chatbot
```

### Future plans
- Using advance model
- Run on GPU with quantized model
- ....