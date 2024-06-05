## complete code in many functions - working fine


import os
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm_name = 'gpt-3.5-turbo-0125'

# Load the document
def load_document():
    loader = DirectoryLoader('data1')
    docs = loader.load()
    return docs

# Split the document
def split_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    return splits

# Debugging: Check the number of splits
def check_splits(splits):
    print(f"Number of document splits: {len(splits)}")

# Embeddings
def create_embeddings(splits):
    embedding = OpenAIEmbeddings()
    return embedding

# Initialize the vector store
def store_in_vector_db(splits, embedding, persist_directory):
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

# Debugging: Check the number of indexed documents
def check_vector_db(vectordb):
    print(f"Number of documents in the vector store: {vectordb._collection.count()}")

# Initialize the LLM
def initialize_llm(llm_name, temperature):
    llm = ChatOpenAI(model_name=llm_name, temperature=temperature)
    return llm

# RetrievalQA chain
def create_retrieval_qa(llm, vectordb):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    return qa_chain

# Test the retrieval
def query_vector_db(qa_chain, query):
    result = qa_chain({"query": query})
    return result["result"]

# Main function
def main():
    # Load the document
    docs = load_document()
    print(docs)

    # Split the document
    splits = split_document(docs)
    check_splits(splits)

    # Embeddings
    embedding = create_embeddings(splits)

    # Initialize the vector store
    vectordb = store_in_vector_db(splits, embedding, 'docs/chroma/')

    # Debugging: Check the number of indexed documents
    check_vector_db(vectordb)

    # Initialize the LLM
    llm = initialize_llm(llm_name, 0)

    # RetrievalQA chain
    qa_chain = create_retrieval_qa(llm, vectordb)

    # Test the retrieval
    query = "Who is the owner of these policies?"
    result = query_vector_db(qa_chain, query)
    print(result)

if __name__ == "__main__":
    main()