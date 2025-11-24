# app.py (snippet)
import os
from dotenv import load_dotenv

# load .env file from project root
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")


########################################################

# app.py
"""
Transcript Q&A Chatbot
- Uses LangChain + OpenAI embeddings + Chat model + Chroma vector DB + Gradio UI
- Expects an .env file with OPENAI_API_KEY and optional CHROMA_PERSIST_DIR, GRADIO_PORT
- Usage: python app.py
"""

import os
import pathlib
import shutil
from dotenv import load_dotenv
from typing import List, Tuple
# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


import gradio as gr
# Tesseract OCR path (update if installed somewhere else)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional: Poppler path for pdf2image
POPPLER_PATH = r"C:\poppler-23.11.0\bin"
from PyPDF2 import PdfReader

# LangChain imports
from langchain.document_loaders.pypdf import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ---------- Gradio ----------
import gradio as gr

# ---------- Load .env ----------
load_dotenv()  # reads .env into environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to your .env file.")

# Optional config - with defaults
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
DEFAULT_COLLECTION = "transcripts"  # used only when you want a default collection name
EMBED_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "1000"))
EMBED_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Ensure persistence dir exists
pathlib.Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path("uploads").mkdir(parents=True, exist_ok=True)

# This dict maps uploaded filename -> chroma collection name (keeps multiple uploads separate)
UPLOAD_TO_COLLECTION = {}


def pdf_has_text_layer(pdf_path: str) -> bool:
    """
    Returns True if PDF contains extractable text (not scanned).
    """
    try:
        reader = PdfReader(pdf_path)
        first_page = reader.pages[0]
        text = first_page.extract_text() or ""
        return len(text.strip()) > 20  # some meaningful text exists
    except:
        return False


def ocr_pdf_to_documents(pdf_path: str) -> List[Document]:
    """
    Convert scanned PDF → images → OCR → LangChain Documents.
    Returns list of Documents (one per page).
    """
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)


    docs = []
    for i, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image, lang="eng")

        # wrap OCR text into a LangChain Document
        docs.append(Document(
            page_content=text,
            metadata={"source": pdf_path, "page": i + 1}
        ))

    return docs


# ---------- Utilities / Backend functions ----------

def load_pdf_to_documents(pdf_path: str) -> List[Document]:
    """
    Load PDF normally if it contains text.
    If scanned / image-only → run OCR.
    """

    if pdf_has_text_layer(pdf_path):
        print("🔍 PDF has text — using PyPDFLoader")
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    else:
        print("🖼️ PDF is scanned — running OCR with Tesseract")
        return ocr_pdf_to_documents(pdf_path)



def split_documents_to_chunks(docs: List[Document], chunk_size: int = EMBED_CHUNK_SIZE, chunk_overlap: int = EMBED_CHUNK_OVERLAP) -> List[Document]:
    """
    Split list of Documents into smaller Document chunks using RecursiveCharacterTextSplitter.
    Returns a list of Document objects (with metadata preserved and chunk index).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    out_docs: List[Document] = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for idx, piece in enumerate(pieces):
            md = dict(d.metadata) if d.metadata else {}
            # store helpful metadata: original source and chunk index
            md.update({"chunk": idx, "source": md.get("source", "")})
            out_docs.append(Document(page_content=piece, metadata=md))
    return out_docs


def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Return an OpenAIEmbeddings instance configured with the API key.
    LangChain will read OPENAI_API_KEY from environment, but we pass it explicitly as a safety.
    """
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def create_and_persist_chroma(docs: List[Document], collection_name: str) -> Chroma:
    """
    Create a Chroma collection from Documents and persist to disk.
    If the collection already exists in the same persist_directory, this will add to it.
    """
    embeddings = get_embeddings_model()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name,
    )
    vectordb.persist()
    return vectordb


def load_chroma_collection(collection_name: str) -> Chroma:
    """
    Load an existing Chroma collection for searches.
    """
    embeddings = get_embeddings_model()
    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectordb


def build_retrieval_qa(vectordb: Chroma, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0) -> RetrievalQA:
    """
    Build a RetrievalQA chain using ChatOpenAI.
    Returns a chain that, given {"query": question}, returns an answer and source docs.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def index_pdf_file(saved_path: str, collection_name: str = None) -> Tuple[str, int]:
    """
    Full pipeline for a single uploaded PDF file:
    - load PDF pages
    - split into chunks
    - create or update Chroma collection named after collection_name (if None derive from filename)
    Returns (collection_name, number_of_chunks)
    """
    if not collection_name:
        collection_name = pathlib.Path(saved_path).stem.replace(" ", "_")

    # 1) load PDF pages
    docs = load_pdf_to_documents(saved_path)

    # 2) split pages into smaller chunks
    chunks = split_documents_to_chunks(docs)

    # 3) create and persist Chroma collection
    create_and_persist_chroma(chunks, collection_name)

    # record mapping
    UPLOAD_TO_COLLECTION[pathlib.Path(saved_path).name] = collection_name

    return collection_name, len(chunks)


def answer_from_collection(collection_name: str, question: str, model_name: str = "gpt-3.5-turbo") -> Tuple[str, List[dict]]:
    """
    Given a collection name and a question, retrieve relevant chunks and ask the LLM.
    Returns (answer_text, list_of_source_metadata).
    """
    vectordb = load_chroma_collection(collection_name)
    qa_chain = build_retrieval_qa(vectordb, model_name=model_name)

    result = qa_chain({"query": question})
    answer_text = result.get("result") or result.get("answer") or ""
    source_docs = result.get("source_documents", [])

    # build a compact list of sources for UI display (page, chunk, short preview)
    sources = []
    for doc in source_docs[:6]:
        md = doc.metadata or {}
        preview = doc.page_content[:400].strip().replace("\n", " ")
        sources.append({
            "metadata": md,
            "preview": preview,
            "length": len(doc.page_content)
        })

    return answer_text, sources


# ---------- Gradio UI ----------

def handle_upload(file, chosen_collection_name: str = None):
    """
    Gradio upload handler.
    - saves uploaded file
    - indexes it into Chroma (creates collection named after the file by default)
    """
    if not file:
        return "No file uploaded.", gr.Dropdown.update(choices=list(UPLOAD_TO_COLLECTION.keys()))

    # save file to uploads folder
    saved_path = os.path.join("uploads", file.name)
    with open(saved_path, "wb") as f:
        f.write(file.read())

    try:
        collection_name, chunk_count = index_pdf_file(saved_path, collection_name=chosen_collection_name)
    except Exception as e:
        return f"Indexing failed: {e}", gr.Dropdown.update(choices=list(UPLOAD_TO_COLLECTION.keys()))

    # update dropdown choices
    choices = list(UPLOAD_TO_COLLECTION.keys())
    status = f"Indexed {file.name} into collection '{collection_name}' ({chunk_count} chunks)."
    return status, gr.Dropdown.update(choices=choices, value=file.name)


def handle_query(selected_file_name: str, question: str, model_name: str):
    """
    Gradio ask handler.
    - loads collection based on selected uploaded filename
    - returns answer and the source previews
    """
    if not selected_file_name:
        return "Please select an uploaded transcript from the left.", ""

    collection_name = UPLOAD_TO_COLLECTION.get(selected_file_name)
    if not collection_name:
        return "Collection not found for that file. Please re-upload.", ""

    try:
        answer_text, sources = answer_from_collection(collection_name, question, model_name=model_name)
    except Exception as e:
        return f"Failed to get answer: {e}", ""

    # format sources for display
    sources_display = ""
    for i, s in enumerate(sources, 1):
        md = s["metadata"]
        page_info = md.get("page") or md.get("page_number") or md.get("source") or ""
        chunk_index = md.get("chunk", "")
        sources_display += f"Source {i}: page={page_info} chunk={chunk_index} length={s['length']}\nPreview: {s['preview']}\n---\n"

    return answer_text, sources_display


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Transcript Q&A — LangChain + OpenAI + Chroma + Gradio")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload & Index")
                file_in = gr.File(label="Upload PDF transcript (PDF)", file_types=[".pdf"])
                custom_collection_name = gr.Textbox(label="Optional: collection name (leave blank to use filename)")
                upload_btn = gr.Button("Upload & Index")
                upload_status = gr.Textbox(label="Upload status", interactive=False)

                gr.Markdown("### Indexed transcripts")
                transcript_dropdown = gr.Dropdown(choices=[], label="Choose indexed transcript (file name)")

                gr.Markdown("⚙️ Model settings")
                model_dropdown = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"], value="gpt-3.5-turbo", label="OpenAI model")
                gr.Markdown("Note: choose a model you have access to in your OpenAI account.")

            with gr.Column(scale=2):
                gr.Markdown("### Ask a question")
                question = gr.Textbox(label="Your question about the transcript", placeholder="e.g. What did Alice say about the deadline?")
                ask_btn = gr.Button("Ask")
                answer_box = gr.Textbox(label="Answer", lines=8)
                sources_box = gr.Textbox(label="Source previews", lines=12)

        # Wire events
        upload_btn.click(fn=handle_upload, inputs=[file_in, custom_collection_name], outputs=[upload_status, transcript_dropdown])
        ask_btn.click(fn=handle_query, inputs=[transcript_dropdown, question, model_dropdown], outputs=[answer_box, sources_box])

        # Footer
        gr.Markdown("**How it works**: upload → index chunks → store embeddings in ChromaDB → ask questions that retrieve relevant chunks → LLM answers using those chunks.")

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False)
