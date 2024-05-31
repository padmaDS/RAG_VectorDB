import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def read_and_split_data(directory='data'):
    # Load environment variables
    
    
    # Load the document
    loader = DirectoryLoader(directory)
    docs = loader.load()
    print(docs)
    
    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    # Debugging: Check the number of splits
    print(f"Number of document splits: {len(splits)}")
    
    return splits

# def create_embeddings_and_store(splits, persist_directory='docs/chroma/'):
#     # Embeddings
#     embedding = OpenAIEmbeddings()
    
#     # Initialize the vector store
#     vectordb = Chroma.from_documents(
#         documents=splits,
#         embedding=embedding,
#         persist_directory=persist_directory
#     )
    
#     # Debugging: Check the number of indexed documents
#     print(f"Number of documents in the vector store: {vectordb._collection.count()}")
    
#     return vectordb

def query_data(persist_directory='docs/chroma/', query="what is the hotel eligibility of an Executive in tier1 actual?"):
    # Load environment variables
    load_dotenv()
    llm_name = 'gpt-3.5-turbo-0125'
    
    # Load the vector store
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )
    
    # Initialize the LLM
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    
    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    
    # Test the retrieval
    result = qa_chain({"query": query})
    
    # Output the result
    print(result["result"])

# Main program
if __name__ == "__main__":
    # Check if the vector database already exists
    persist_directory = 'docs/chroma/'
    if not os.path.exists(persist_directory):
        splits = read_and_split_data()
        # create_embeddings_and_store(splits, persist_directory)
    
    query_data(persist_directory)
