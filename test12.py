## complete code at one place 

import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm_name = 'gpt-4'


# Load the document
loader = DirectoryLoader('data1')
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

# Embeddings
embedding = OpenAIEmbeddings()

# Initialize the vector store
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# Debugging: Check the number of indexed documents
print(f"Number of documents in the vector store: {vectordb._collection.count()}")

# Initialize the LLM
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# Test the retrieval
query = "what is the compensatory leave?"
result = qa_chain({"query": query})

# Output the result
print(result["result"])

