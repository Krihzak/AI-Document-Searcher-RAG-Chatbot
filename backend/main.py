import os
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


app = FastAPI(
    title="RAG Chatbot API",
    description="API for handling document uploads and chat for the RAG chatbot.",
    version="1.0.0"
)

#CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, openai_api_key]):
        raise ValueError("API keys for Pinecone or OpenAI are not set in the .env file.")

    # Initialize the Pinecone client using the new v3 SDK
    from pinecone import Pinecone as PineconeNativeClient
    pc = PineconeNativeClient(api_key=pinecone_api_key)
    
    INDEX_NAME = "rag-chatbot-index"

    # Check if the index exists
    if INDEX_NAME not in pc.list_indexes().names():
         raise ValueError(f"Pinecone index '{INDEX_NAME}' not found. Please create it first.")

    # Initialize LangChain components
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # This is the LangChain Pinecone integration, which is different from the native client
    vectorstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

except Exception as e:
    print(f"Error initializing services: {e}")
    raise

# API Endpoints 

@app.get("/", tags=["Status"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "RAG Chatbot API is running!"}

@app.post("/upload", tags=["Document Handling"])
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF, processes it, splits it into chunks,
    generates embeddings, and stores them in Pinecone.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 1. Load the document using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 2. Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Create embeddings and store them in Pinecone
        vectorstore.add_documents(docs)

        return {"message": f"Successfully uploaded and processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat", tags=["Chat"])
async def chat_with_doc(query: str = Form(...)):
    """
    Takes a user query, performs a similarity search in the vector store,
    and uses an LLM to generate a response based on the retrieved context.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Perform a similarity search to find relevant document chunks
        relevant_docs = vectorstore.similarity_search(query, k=3)

        if not relevant_docs:
            return {"answer": "I couldn't find any relevant information in the document to answer your question."}

        # Use the LangChain QA chain to get the answer
        answer = chain.run(input_documents=relevant_docs, question=query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {str(e)}")

