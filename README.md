# 🤖 PDF Chatbot with OCR (Tesseract + LangChain)

A lightweight Python-based chatbot that reads PDFs — including scanned ones — and answers questions about their content. It uses LangChain, OpenAI, Chroma, and Tesseract OCR to extract, index, and retrieve information from uploaded documents.

---

## 🚀 Features

- Extract text from:
  - Digital/searchable PDFs
  - Scanned PDFs using OCR (Tesseract)
- Converts scanned pages to images (pdf2image)
- Chunking + embeddings + Chroma vector storage
- Question-answering over PDF content
- Simple and clean Gradio UI

---

## 🧩 Tech Stack

- **LangChain** – text splitting, embeddings, RetrievalQA  
- **OpenAI API** – language model for answers  
- **ChromaDB** – vector database  
- **Tesseract OCR** – scanned PDF text extraction  
- **pdf2image + Pillow** – PDF → image conversion  
- **Gradio** – user interface  

---

## 📦 Installation


```bash
### 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

### 2. Create a virtual environment
python -m venv venv

### 3. Install dependencies
pip install -r requirements.txt

### 4. Install external tools
Tesseract OCR
Poppler
Add both installations to your system PATH.

### 5. Running the App
python app.py
