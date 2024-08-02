from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Create vector db from context
class ContextVectorDB:
    def __init__(self) -> None:
        self.vector_db_path = "vectorStores/db_faiss"
        self.model_file = "models/all-MiniLM-L6-v2-f16.gguf"
    
    def create_db_from_text(self, raw_text: str):
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 512,
            chunk_overlap = 50,
            length_function = len
        )
        
        chunks = text_splitter.split_text(raw_text)
        
        # đưa đoạn text đã split vào embedding
        gpt4all_kwargs = {'allow_download': 'True'}  
        embedding_model = GPT4AllEmbeddings(model_file = self.model_file, gpt4all_kwargs = gpt4all_kwargs)
        
        # dua vao Faiss Vector DB
        db = FAISS.from_texts(
            texts = chunks,
            embedding = embedding_model
        )
        db.save_local(self.vector_db_path)

        return db

# create vector db from file
class FileVectorDB:
    def __init__(self) -> None:
        self.pdf_data_path = "data"
        self.vector_db_path = "vectorStores/db_faiss"
        self.model_file = "models/all-MiniLM-L6-v2-f16.gguf"
        
    def create_db_from_files(self, pdf_data_path: str):
        loader = DirectoryLoader(self.pdf_data_path, glob = pdf_data_path, loader_cls = PyPDFLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 50
        )
        chunks = text_splitter.split_documents(documents)
        
        # đưa đoạn text đã split vào embedding 
        gpt4all_kwargs = {'allow_download': 'True'}  
        embedding_model = GPT4AllEmbeddings(model_file = self.model_file, gpt4all_kwargs = gpt4all_kwargs)
        
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(self.vector_db_path)
        return db