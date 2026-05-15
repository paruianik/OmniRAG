from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredImageLoader, UnstructuredPDFLoader
import whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional
import tempfile
from dotenv import load_dotenv
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "server is running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(
    upload_type: str = Form(...),
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    pdf_path = None
    try:
        # ========================
        # 1. LOAD BASED ON TYPE
        # ========================
        if upload_type == "PDF":
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            pdf_path = temp_file.name
            loader = UnstructuredPDFLoader(pdf_path)
            data = loader.load()

        elif upload_type == "Image":
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded")
            
            # Get correct extension from uploaded file
            ext = os.path.splitext(file.filename)[-1] or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            pdf_path = temp_file.name  # reusing variable for cleanup
            loader = UnstructuredImageLoader(pdf_path)
            data = loader.load()

        elif upload_type == "Video":
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded")
            
            # Save video to temp file
            ext = os.path.splitext(file.filename)[-1] or ".mp4"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            pdf_path = temp_file.name  # reusing variable for cleanup

            # Transcribe audio using Whisper
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(pdf_path)
            transcript = result["text"]

            # Wrap transcript as a document
            from langchain_core.documents import Document
            data = [Document(page_content=transcript)]

        elif upload_type == "URL":
            if not url:
                raise HTTPException(status_code=400, detail="No URL provided")
            
            loader = UnstructuredURLLoader(urls=[url])
            data = loader.load()

        else:
            raise HTTPException(status_code=400, detail="Invalid upload type")

        # ========================
        # 2. SPLIT + EMBED + STORE
        # ========================
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file")

        embeddings = HuggingFaceEmbeddings()
        vectorstore = Chroma.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # ========================
        # 3. RAG CHAIN
        # ========================
        #llm = Ollama(model="phi4-mini")
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

        prompt_template = """
        You are a factual assistant.
        Answer the question ONLY using the context below.
        
        - Do NOT use prior knowledge
        - Do NOT refuse unless context is empty
        - If answer exists in context, give it directly
        - Give full length answer
        
        Context:
        {context}
        Question:
        {question}
        Answer:
        """

        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        return {"answer": answer}

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file if it was created
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
