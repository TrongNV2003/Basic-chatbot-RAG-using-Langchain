from langchain_community.llms import ctransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit
import torch

class Chatbot:
    def __init__(self):
        self.model_file = "models/vinallama-7b-chat_q5_0.gguf"
        self.vector_db_path = "VectorStores"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_file.to(self.device)
        
    def load_llm(self, model_file):
        llm = ctransformers(
            model = model_file,
            model_type = "llama",
            max_new_tokens = 1024,
            temperature = 0.01
        )
        return llm

    def create_prompt(self, template):
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return prompt

    def create_qa_chain(self, prompt, llm, db):
        llm_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = db.as_retriever(search_kwargs = {"k": 3}, max_new_tokens = 1024),
            return_source_documents = False,
            chain_type_kwargs = {"prompt": prompt}
        )
        return llm_chain

    def read_vector_db(self):
        model_file = "models/all-MiniLM-L6-v2-f16.gguf"  
        gpt4all_kwargs = {'allow_download': 'True'}  

        embedding_model = GPT4AllEmbeddings(model_file = model_file, gpt4all_kwargs = gpt4all_kwargs)
        db = FAISS.load_local(self.vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    
    def format_response(self, response):
        user_query = response.get('query', 'N/A')
        bot_response = response.get('result', 'N/A')
        
        formatted_output = f"User: {user_query}\nbotgpt: {bot_response}"
        return formatted_output
    
    def runbot(self, question: str, prompt: str):
        db = self.read_vector_db()
        llm = self.load_llm(self.model_file)
        prompting = self.create_prompt(prompt)
        llm_chain = self.create_qa_chain(prompting, llm, db)
        
        response = llm_chain.invoke({"query": question})
        formatted_output = self.format_response(response)
        
        return formatted_output