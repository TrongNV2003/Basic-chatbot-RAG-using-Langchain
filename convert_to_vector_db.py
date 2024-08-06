from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Create vector db from context
class ContextVectorDB:
    def __init__(self) -> None:
        self.vector_db_path = "VectorStores"
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
        self.vector_db_path = "VectorStores"
        self.model_file = "models/all-MiniLM-L6-v2-f16.gguf"
        
    def create_db_from_files(self, pdf_file: str):
        loader = DirectoryLoader(self.pdf_data_path, glob = pdf_file, loader_cls = PyPDFLoader)
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
    
if __name__ == "__main__":
# convert a context to vectorDB
    # context_db = ContextVectorDB()
    # context = "Thép đã tôi thế đấy không phải là một tác phẩm văn học chỉ nhìn đời mà viết. Tác giả sống nó rồi mới viết nó. Nhân vật trung tâm Pa-ven chính là tác giả: Nhi-ca-lai A-xtơ-rốp- xki. Là một chiến sĩ cách mạng tháng Mười, ông đã sống một cách nồng cháy nhất, như nhân vật Pa-ven của ông. Cũng không phải một cuốn tiểu thuyết tự thuật thường vì hứng thú hay lợi ích cá nhân mà viết. A-xtơ-rốp-xki viết Thép đã tôi thế đấy trên giường bệnh, trong khi bại liệt và mù, bệnh tật tàn phá chín phần mười cơ thể. Chưa bao giờ có một nhà văn sáng tác trong những điều kiện gian khổ như vậy. Trong lòng người viết phải có một nhiệt độ cảm hứng nồng nàn không biết bao nhiêu mà kể. Nguồn cảm hứng ấy là sức mạnh tinh thần của người chiến sĩ cách mạng bị tàn phế, đau đớn đến cùng cực, không chịu nằm đợi chết, không thể chịu được xa rời chiến đấu, do đó phấn đấu trở thành một nhà văn và viết nên cuốn sách này. Càng yêu cuốn sách, càng kính trọng nhà văn, càng tôn quí phẩm chất của con người cách mạng."
    # context_db.create_db_from_text(context)


# convert all file pdf to vectorDB
    file_db = FileVectorDB()
    pdf_file = "1.pdf"
    file_db.create_db_from_files(pdf_file)