# RAG Chatbot for the BS in Data Science Program

This project is an intelligent, retrieval-augmented generation (RAG) chatbot designed to answer questions about the BS in Data Science degree program. It serves as a helpful assistant for students, providing accurate information based on a custom knowledge base of official course documents and a student handbook.

This project was built as the final assignment for Module 1 of the Ready Tensor AI Agentic AI Developer Certification.

## ✨ Features

* **Intelligent Routing:** The agent analyzes the user's question to determine if it's a general question for the handbook or a content-specific question about a particular subject.
* **Hybrid Knowledge Base:**
    * Uses a persistent **ChromaDB vector store** for efficient semantic search on the large student handbook.
    * Maintains a **JSON database** of full subject documents to provide complete, high-context answers for specific course-related queries.
* **High-Speed Generation:** Powered by the **Groq API** (`Llama 3.1`) for fast, near-instantaneous LLM responses.
* **Local Embeddings:** Utilizes a local `HuggingFace sentence-transformer` model for cost-free and private text embedding.
* **Multiple Interfaces:** Can be run as an interactive command-line tool or as a web-based chat interface.

## 📂 Project Structure

```
.
├── config/
│   └── rag_prompts.yaml    # Contains all prompt templates for the router and final answer
├── data/
│   └── degree_data/        # Folder for all the .txt source documents
├── handbook_db/            # The persistent ChromaDB vector store for the handbook
├── subjects_db.json        # The JSON file containing the full text of all subject files
├── ingest.py               # Script to process documents and build the knowledge bases
├── rag.py                  # The command-line (terminal) chatbot application
├── app.py                  # The Gradio web interface application
├── .env                    # File for storing secret API keys
└── README.md               # This file
```

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys**
    Create a file named `.env` in the root of the project directory and add your Groq API key:
    ```
    GROQ_API_KEY="gsk_YourActualApiKeyHere"
    ```

## 🚀 Usage

Before running either application, you must first build the knowledge base.

**Step 1: Run the Ingestion Script (Run This Once)**
This script will process all your documents and create the `handbook_db` folder and the `subjects_db.json` file.
```bash
python ingest.py
```

Once ingestion is complete, you can use the chatbot in one of three ways:

---
#### **Option 1: Run as a Command-Line Chatbot**

This will start the interactive chat session directly in your terminal.
```bash
python rag.py
```
To end the chat, type `exit`.

---
#### **Option 2: Run the Web Interface Locally**

This will launch a local web server with a user-friendly chat interface.
```bash
python app.py
```
After running the command, your terminal will show a local URL (like `http://127.0.0.1:7860`). Open this URL in your web browser to use the chatbot.

---
#### **Option 3: Use the Deployed Public URL**

Once the project is deployed on Hugging Face Spaces, you can access the chatbot directly via its public URL. No installation is required.

**URL:** `https://huggingface.co/spaces/Honey1811/bs-degree-chatbot`

---

## 🛠️ Technology Stack

* **Core Framework:** LangChain
* **Web UI:** Gradio
* **LLM:** Groq (Llama 3.1)
* **Vector Database:** ChromaDB
* **Embedding Model:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
* **Configuration:** PyYAML, python-dotenv
* **Data Validation:** Pydantic