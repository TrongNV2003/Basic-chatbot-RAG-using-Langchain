# Recommend execute on google colab

import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

chat_history = []

folder_path = "db"

# MODEL_QUANTIZED = "Trongdz/Llama-1.1B-AWQ-4bit"   # if run on google colab
MODEL_QUANTIZED = "models/Llama-1.1B-AWQ-4bit"      # else

model = AutoAWQForCausalLM.from_pretrained(MODEL_QUANTIZED, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_QUANTIZED, use_fast=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
    """
    <|im_start|>user
    You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so.
    Context:{context}<|im_end|>\n<|im_start|>assistant\n
    """
)


def ai_query(query):
    print(f"Question: {query}")

    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=1024)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)

    response_answer = {"answer": response}
    return response_answer

# đang bị lỗi
def rag_pdf_query(query):
    print(f"Question: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generate a search query to lookup in order to get information relevant to the conversation",
            ),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=retriever_prompt
    )

    document_chain = create_stuff_documents_chain(model, raw_prompt)
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    result = retrieval_chain.invoke({"input": query})
    print(result["answer"])
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


def pdf_upload(file_path):
    print(f"Uploading file: {file_path}")

    loader = PDFPlumberLoader(file_path)
    docs = loader.load_and_split()
    print(f"Number of documents loaded: {len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": os.path.basename(file_path),
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def main():
    while True:
        print("Select an option:")
        print("1. Ask a general AI query")
        print("2. Ask a PDF-related query")
        print("3. Upload a PDF")
        print("4. Exit")
        
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            query = input("Enter your query: ")
            response = ai_query(query)
            print(f"Response: {response['answer']}")
        elif choice == "2":
            query = input("Enter your query: ")
            response = rag_pdf_query(query)
            print(f"Response: {response['answer']}")
            print("Sources:")
            for source in response["sources"]:
                print(f"- {source['source']}: {source['page_content']}")
        elif choice == "3":
            file_path = input("Enter the path to the PDF file: ")
            response = pdf_upload(file_path)
            print(f"Response: {response}")
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
    
    
# Using Faiss
  
# import os
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.document_loaders import PDFPlumberLoader
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# import torch
# from transformers import AutoTokenizer
# from awq import AutoAWQForCausalLM

# chat_history = []

# folder_path = "db"

# MODEL_QUANTIZED = "Trongdz/Llama-1.1B-AWQ-4bit"   # if run on google colab
# # MODEL_QUANTIZED = "models/Llama-1.1B-AWQ-4bit"      # else

# model = AutoAWQForCausalLM.from_pretrained(MODEL_QUANTIZED, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_QUANTIZED, use_fast=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
# )

# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
#     [INST] {input}
#            Context: {context}
#            Answer:
#     [/INST]
# """
# )


# def ai_query(query):
#     print(f"Query: {query}")

#     input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=1024)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     print(response)

#     response_answer = {"answer": response}
#     return response_answer


# def rag_pdf_query(query):
#     print(f"Query: {query}")

#     print("Loading vector store")
#     vector_store = FAISS.load_local(folder_path, embedding, allow_dangerous_deserialization=True)

#     print("Creating chain")
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 10,
#             "score_threshold": 0.3,
#         },
#     )

#     retriever_prompt = ChatPromptTemplate.from_messages(
#         [
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#             (
#                 "human",
#                 "Given the above conversation, generate a search query to lookup in order to get information relevant to the conversation",
#             ),
#         ]
#     )

#     history_aware_retriever = create_history_aware_retriever(
#         llm=model, retriever=retriever, prompt=retriever_prompt
#     )

#     document_chain = create_stuff_documents_chain(model, raw_prompt)
#     retrieval_chain = create_retrieval_chain(
#         history_aware_retriever,
#         document_chain,
#     )

#     result = retrieval_chain.invoke({"input": query})
#     print(result["answer"])
#     chat_history.append(HumanMessage(content=query))
#     chat_history.append(AIMessage(content=result["answer"]))

#     sources = []
#     for doc in result["context"]:
#         sources.append(
#             {"source": doc.metadata["source"], "page_content": doc.page_content}
#         )

#     response_answer = {"answer": result["answer"], "sources": sources}
#     return response_answer


# def pdf_upload(file_path):
#     print(f"Uploading file: {file_path}")

#     loader = PDFPlumberLoader(file_path)
#     docs = loader.load_and_split()
#     print(f"Number of documents loaded: {len(docs)}")

#     chunks = text_splitter.split_documents(docs)
#     print(f"Number of chunks: {len(chunks)}")

#     vector_store = FAISS.from_documents(chunks, embedding)
#     vector_store.save_local(folder_path)

#     response = {
#         "status": "Successfully Uploaded",
#         "filename": os.path.basename(file_path),
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }
#     return response


# def main():
#     while True:
#         print("Select an option:")
#         print("1. Ask a general AI query")
#         print("2. Ask a PDF-related query")
#         print("3. Upload a PDF")
#         print("4. Exit")
        
#         choice = input("Enter your choice: ").strip()

#         if choice == "1":
#             query = input("Enter your query: ")
#             response = ai_query(query)
#             print(f"Response: {response['answer']}")
#         elif choice == "2":
#             query = input("Enter your query: ")
#             response = rag_pdf_query(query)
#             print(f"Response: {response['answer']}")
#             print("Sources:")
#             for source in response["sources"]:
#                 print(f"- {source['source']}: {source['page_content']}")
#         elif choice == "3":
#             file_path = input("Enter the path to the PDF file: ")
#             response = pdf_upload(file_path)
#             print(f"Response: {response}")
#         elif choice == "4":
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice. Please try again.")


# if __name__ == "__main__":
#     main()

