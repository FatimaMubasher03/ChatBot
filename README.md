# 🎤 Transcript Q&A Chatbot (Python + LangChain + ChromaDB)

This project is a local, privacy-friendly Q&A Chatbot that lets you upload a PDF transcript, extracts text using PyPDF, splits the content into chunks, stores them in ChromaDB, and answers user questions based on the transcript — with OCR support planned for the next version.

## 🚀 Features
- Upload a PDF transcript
- Automatic text extraction using:
  - **PyPDF** (normal PDFs)
  - **OCR (Tesseract)** for scanned PDFs/images
- Chunking & text splitting with LangChain
- Embeddings stored in **ChromaDB**
- Fast semantic search for relevant answers
- Chat-style Q&A interface
- Cleaned & indexed transcript storage
- Environment variables supported via `.env`

---

## 📂 Project Structure
ChatBot/
│── app.py
│── requirements.txt
│── .gitignore
│── uploads/ # Uploaded PDFs
│── chroma_db/ # Vector database
│── venv/ # Virtual environment (ignored)
│── .env # API keys or secrets (ignored)


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/FatimaMubasher03/ChatBot
cd ChatBot

2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the App
python app.py


Your chatbot is now ready for Q&A magic ✨

🧠 How It Works
User uploads a PDF
PyPDF extracts text
Text is split into chunks using recursive splitters
ChromaDB stores embeddings
User asks questions
System retrieves relevant chunks + LLM generates the answer
