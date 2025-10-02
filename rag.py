import os
import json
import yaml
from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate



load_dotenv()

# DEFINE PATHS TO OUR DATABASE AND CONFIG
HANDBOOK_DB_PATH = "./handbook_db"
SUBJECTS_DB_PATH = "subjects_db.json"
PROMPT_CONFIG_PATH = "config/rag_prompts.yaml"


def load_json_db(file_path: str) -> Dict[str, Any]:
    """
    Loads the subjects json database from specified path.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration files.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)


print("Initializing components... Please wait.")

subjects_db = load_json_db(SUBJECTS_DB_PATH)

prompt_configs = load_yaml_config(PROMPT_CONFIG_PATH)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

handbook_client = chromadb.PersistentClient(path=HANDBOOK_DB_PATH)
handbook_collection = handbook_client.get_collection(name="handbook")

llm = ChatGroq(model="llama-3.1-8b-instant")

print("Initialisation Complete. All components are ready.")
print("-" * 50)


#  ----  LLM ROUTER LOGIC ----

class RouterOutput(BaseModel):
    """
    Defines the structured output for the router's decision.
    """
    query_type: str = Field(description="The type of query. Either 'subject_content' or 'general_handbook_query'.")
    subjects: List[str] = Field(description="A list of specific subject keyword found in user's question. Should be an empty list if query_type is 'general_handbook_query'.")


def get_router_decision(user_question: str, prompt_configs: Dict) -> Dict:
    """
    Uses an LLM to classify the user's question and extract subject keywords.
    """
    structured_llm = llm.with_structured_output(RouterOutput)

    subject_keywords = list(subjects_db.keys())

    prompt_template_text = prompt_configs['router_prompt']
    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["user_question", "subject_keywords"]
    )

    router_chain = prompt | structured_llm

    decision = router_chain.invoke({
        "user_question" : user_question,
        "subject_keywords" : subject_keywords
    })

    return decision



#   ------ CONTEXT RETRIEVAL LOGIC -----
def retrieve_context(user_question: str, decision: RouterOutput) -> str:
    """
    Retrieves the appropriate context based on the router's decision.
    """
    query_type = decision.query_type
    subjects = list(set(decision.subjects))

    print(f"Router decided query type is {query_type}")

    if query_type == "subject_content":
        print(f"Retrieving content for subject(s): {subjects}")
        context = ""
        for subject_key in subjects:
            subject_content = subjects_db.get(subject_key)
            if subject_content:
                context += subject_content+"\n\n"

        if not context:
            return "Could not find the the specified subject document."
        
    else:
        print("Performing vector search on the handbook...")

        query_vector = embedding_model.embed_query(user_question)

        results = handbook_collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=["documents"]
        )
        context = "\n\n----\n\n".join(results["documents"][0])
    
    return context



#   ---- MAIN EXECUTION BLOCK WITH INTERACTIVE CHAT -----
if __name__ == "__main__":
    print("AI Assistant: I'm ready! Ask me anything about your query.")
    print("               (Type 'exit' to end the chat)")
    print("-" * 50)

    while True:
        user_question = input("You: ")

        if user_question.lower() == "exit":
            print("AI Assistant: GoodBye! ")
            break

        decision = get_router_decision(user_question, prompt_configs)

        context = retrieve_context(user_question, decision)

        final_prompt_template = prompt_configs['rag_final_prompt']

        prompt = PromptTemplate(
            template=final_prompt_template,
            input_variables=["context","question"]
        )

        final_chain = prompt | llm

        ai_response = final_chain.invoke({
            "context": context,
            "question": user_question
        })

        print(f"\nAI Assistant:\n{ai_response.content}\n")
        print("-" * 150 )