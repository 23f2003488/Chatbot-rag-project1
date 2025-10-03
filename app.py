import os
import json
import yaml
import gradio as gr
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

# --- 1. SETUP PHASE (Done only once when the app starts) ---
print("AI Assistant: Initializing... This may take a moment.")
load_dotenv()


HANDBOOK_DB_PATH = "./handbook_db"
SUBJECTS_DB_PATH = "subjects_db.json"
PROMPT_CONFIG_PATH = "config/rag_prompts.yaml"


def load_json_db(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load databases and configs
subjects_db = load_json_db(SUBJECTS_DB_PATH)
prompt_configs = load_yaml_config(PROMPT_CONFIG_PATH)


# Initialize models and database connections
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
handbook_client = chromadb.PersistentClient(path=HANDBOOK_DB_PATH)
handbook_collection = handbook_client.get_collection(name="handbook")
llm = ChatGroq(model="llama-3.1-8b-instant")


# Pydantic model for the router
class RouterOutput(BaseModel):
    query_type: str = Field(description="The type of query. Either 'subject_content' or 'general_handbook_query'.")
    subjects: List[str] = Field(description="A list of specific subject keyword found in user's question.")

print("AI Assistant: Initialization complete. Ready for interaction.")




# --- 2. CORE RAG LOGIC  ---

def get_router_decision(user_question: str) -> RouterOutput:
    structured_llm = llm.with_structured_output(RouterOutput)
    subject_keywords = list(subjects_db.keys())
    prompt_template_text = prompt_configs['router_prompt']
    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["user_question", "subject_keywords"]
        )
    router_chain = prompt | structured_llm
    return router_chain.invoke({
        "user_question": user_question,
        "subject_keywords": subject_keywords
        })


def retrieve_context(user_question: str, decision: RouterOutput) -> str:
    query_type = decision.query_type
    subjects = list(set(decision.subjects))

    if query_type == "subject_content":
        context = ""
        for subject_key in subjects:
            subject_content = subjects_db.get(subject_key)
            if subject_content:
                context += subject_content + "\n\n"
        return context if context else "Could not find the specified subject document."
    else:
        query_vector = embedding_model.embed_query(user_question)
        results = handbook_collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=["documents"]
            )
        return "\n\n---\n\n".join(results["documents"][0])



# --- 3. THE MAIN CHAT FUNCTION FOR GRADIO ---
def chat_with_agent(user_question, history):
    """
    This is the main function that Gradio's ChatInterface will call.
    It takes the user's question, and returns the AI's response.
    """
    print(f"User Query: {user_question}")

    # 1. Route
    decision = get_router_decision(user_question)
    print(f"Router Decision: {decision.query_type}, Subjects: {decision.subjects}")

    # 2. Retrieve
    context = retrieve_context(user_question, decision)

    # 3. Generate
    final_prompt_template = prompt_configs['rag_final_prompt']
    prompt = PromptTemplate(
        template=final_prompt_template, 
        input_variables=["context", "question"]
        )
    final_chain = prompt | llm
    ai_response = final_chain.invoke({
        "context": context, 
        "question": user_question
        })
    
    return ai_response.content




# --- 4. CREATE AND LAUNCH THE GRADIO INTERFACE ---
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="BS in Data Science - RAG Assistant",
    description="Ask me any question about the BS in Data Science degree program, its courses, or its rules.",
    examples=[
        "What will I learn in the Computational Thinking course?",
        "What are the rules for the final exam?",
        "Tell me about the foundational level subjects"
    ]
)


if __name__ == "__main__":
    demo.launch() 