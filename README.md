# RAG Chatbot for the BS in Data Science Program

This project is an intelligent, retrieval-augmented generation (RAG) chatbot designed to answer questions about the BS in Data Science degree program. It serves as a helpful assistant for students, providing accurate information based on a custom knowledge base of official course documents and a student handbook.

This project was built as the final assignment for Module 1 of the Ready Tensor AI Agentic AI Developer Certification.

## ✨ Features

* **Intelligent Routing:** The agent analyzes the user's question to determine if it's a general question for the handbook or a content-specific question about a particular subject.
* **Hybrid Knowledge Base:**
    * Uses a persistent **ChromaDB vector store** for efficient semantic search on the large student handbook.
    * Maintains a **JSON database** of full subject documents to provide complete, high-context answers for specific course-related queries.
* **High-Speed Generation:** Powered by the **Groq API** (`Llama 3.1 8B`) for fast, near-instantaneous LLM responses.
* **Local Embeddings:** Utilizes a local `HuggingFace sentence-transformer` model for cost-free and private text embedding.
* **Interactive CLI:** A simple and user-friendly command-line interface for chatting with the agent.
* **Modular & Robust:** Built with a modular architecture, separating data ingestion from the main RAG application, and includes hardened prompts to ensure reliability and security.

## 📂 Project Structure

```
.
├── config/
│   └── rag_prompts.yaml    # Contains all prompt templates for the router and final answer
├── data/
│   └── degree_data/        # Folder for all the .txt source documents (handbook, subjects)
├── handbook_db/            # The persistent ChromaDB vector store for the handbook
├── subjects_db.json        # The JSON file containing the full text of all subject files
├── ingest.py               # Script to process documents and build the knowledge bases
├── rag.py                  # The main interactive chatbot application
├── .env                    # File for storing secret API keys
└── README.md               # This file
```

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**2. Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
This project requires several Python packages.
```bash
pip install -r requirements.txt
```
*(Note: If you don't have a `requirements.txt` file yet, you can create one by running `pip freeze > requirements.txt` in your terminal after installing the packages below.)*
```bash
pip install langchain langchain-groq langchain-huggingface chromadb pydantic python-dotenv pyyaml sentence-transformers
```

**4. Set Up API Keys**
Create a file named `.env` in the root of the project directory and add your Groq API key:
```
GROQ_API_KEY="gsk_YourActualApiKeyHere"
```

## 🚀 Usage

The application is split into two main scripts: `ingest.py` (run once) and `rag.py` (run anytime).

**1. Place Your Data**
Make sure all your `.txt` files (the handbook and all subject files) are located inside the `data/degree_data` folder.

**2. Run the Ingestion Script (Run This Once)**
This script will process all your documents and create the `handbook_db` and `subjects_db.json`.
```bash
python ingest.py
```

**3. Run the RAG Chatbot**
Once ingestion is complete, you can start the interactive chatbot.
```bash
python rag.py
```
The application will load the knowledge bases and prompt you to start asking questions. To end the chat, simply type `exit`.

## 🛠️ Technology Stack

* **Core Framework:** LangChain
* **LLM:** Groq (Llama 3.1 8B)
* **Vector Database:** ChromaDB
* **Embedding Model:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
* **Configuration:** PyYAML, python-dotenv
* **Data Validation:** Pydantic