from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import os
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = api_key

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load documents from directory
directory = 'data1'
loader = DirectoryLoader(directory)
data = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# Create an ephemeral Chroma client
new_client = chromadb.EphemeralClient()

# Initialize Chroma vector store with the documents and embeddings
openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=new_client, collection_name="openai_collection"
)

# Initialize OpenAI language model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Set up the retriever
retriever = openai_lc_client.as_retriever()

# Define the prompt template
template = """You are an Intelligent CosPower AI Assistant.
You are designed to respond to answers as per the input language regarding the HR Leaves and travel policies.
You are primarily programmed to communicate in English. However, if user asks in another language,
you must strictly respond in the same language as the user’s language. Do not respond in English for other language queries.

{context}

Question: {question}
Helpful Answer:"""
rag_prompt = PromptTemplate.from_template(template)

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# Initialize Flask app
app = Flask(__name__)

# Function to query the RAG chain
def gg(question):
    response = rag_chain.invoke(question)
    return response.content

# Define your API endpoint
@app.route('/cospower_multi', methods=['POST'])
def query_endpoint():
    # Get the question from the request
    question = request.json.get('query')

    # Invoke the RAG chain with the question
    answer = gg(question)

    # Return the answer as JSON response
    return jsonify({"answer": answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
