## complete code with 2 simple functions

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Function to create the vector store
def create_vector_store(documents_directory):
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Load the document
    loader = DirectoryLoader(documents_directory)
    docs = loader.load()

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embedding = OpenAIEmbeddings()

    # Initialize the vector store
    persist_directory = 'docs/chroma/'
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

# Function to query the vector store
def query_vector_store(query, vector_store, llm_name='gpt-3.5-turbo-0125'):
    # Initialize the LLM
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever()
    )

    # Test the retrieval
    result = qa_chain({"query": query})

    return result["result"]

# Example usage
if __name__ == "__main__":
    documents_directory = 'data1'
    vector_store = create_vector_store(documents_directory)
    query = "who is the owner of these policies?"
    result = query_vector_store(query, vector_store)
    print(result)
