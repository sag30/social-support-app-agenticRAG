import os
import json,re
import subprocess
import argparse
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from services.fastapi_service.recommendation_api import get_recommendations
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

current_applicant = None

def extract_json_block(text):
    """Extract the first JSON object from text using regex. Returns dict or None."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def etl_agent(_: str) -> dict:
    try:
        # 1. Extraction
        subprocess.run(["python", "etl_pipeline.py"], check=True)
        # 2. Feature engineering
        subprocess.run(["python", "feature_engineering.py"], check=True)
        # 3. DB ingest
        subprocess.run(["python", "db_ingest.py"], check=True)
        # 4. Chroma ingest
        subprocess.run(["python", "chroma_ingest.py"], check=True)

        # Check manifest
        manifest_path = "data/processed/manifest.json"
        if not os.path.exists(manifest_path):
            return {"error": "No manifest.json generated; no files processed."}
        with open(manifest_path, "r") as f:
            content = f.read()
            if not content.strip():
                return {"error": "Empty manifest.json; no files processed."}
    except Exception as e:
        return {"error": f"ETL failed: {e}"}


def model_agent(applicant_key: str) -> dict:
    """
    Micro-agent: call recommendation API for eligibility.
    """
    eligibility_check= get_recommendations(applicant_key)
    print("Shruti: In model_agent(), Eligibility check result:", eligibility_check)
    return eligibility_check


def rag_agent(question: str, applicant_key: str) -> dict:
    """
    Micro-agent: perform RAG retrieval for given question using Ollama/mistral.
    """
    # Use Ollama for both embeddings and LLM (if Ollama server is running and mistral is pulled)
    emb = OllamaEmbeddings(model="mistral")  
    vectordb = Chroma(
        persist_directory="chromadb_data",
        embedding_function=emb,
        collection_name="support_documents"
    )
    # Filter by applicant_key if supported in your pipeline (add filter logic if needed)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4},
                                      filters={"applicant_key": {"$eq": applicant_key}}
                                      )

    llm = ChatOllama(model="mistral")  # Adjust to your local model's name if needed

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    out = qa.invoke({"query": question})
    # Make sure to return as dict for agent chaining
    if isinstance(out, dict) and "result" in out:
        return {"rag_answer": out.get("result",out)}
    else:
        return {"rag_answer": str(out)}
    
# Define tools
tools = [
    Tool(
        name="ETL",
        func=lambda arg: json.dumps(etl_agent(arg)),
        description="Run the ETL pipeline; ignore input value."
    ),
    Tool(
        name="EligibilityModel",
        func=lambda key: json.dumps(model_agent(key)),
        description="Get program recommendations for applicant_key."
    ),
    Tool(
        name="RAGRetrieval",
        func=lambda q: rag_agent(q, current_applicant),
        description="Answer questions about the applicant's documents."
    )
]

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a master orchestrator AI agent. You will receive a JSON string as input, with applicant_key and question fields.


You have the following tools: ETL, EligibilityModel, RAGRetrieval.
Use the following format:

Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
Observation: <tool output>
... (repeat Thought/Action as needed) ...
Thought: <final reasoning>
Final Answer: <**a JSON object** with keys "manifest", "recommendations", "rag_answer">

Never add explanations, markdown, 'Final Answer:', 'Observation', or extra text — ONLY the JSON object as output.

Example output:
{
  "manifest": { "processed_files": ["file1.pdf", "file2.pdf"] },
  "recommendations": { "eligible": true, "recommendations": {"Upskilling Grant": 1.0, ...}},
  "rag_answer": "The account number is AE987654321098765432109."
}

Input: {input}
"""
)

# Initialize agent
agent_executor = initialize_agent(
    tools,
    ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    prompt=prompt,
    handle_parsing_errors=True
)

def run_master(applicant_key: str, question: str = None) -> dict:
    global current_applicant
    current_applicant = applicant_key
    input_text = {
        "applicant_key": applicant_key,
        "question": question or ""
    }
    try:
        output = agent_executor.run({"input": json.dumps(input_text)})
        # Ensure JSON-only output
        json_text = extract_json_block(output)
        if not json_text:
            # Fallback: wrap raw as rag_answer
            return {"manifest": None, "recommendations": None, "rag_answer": output.strip()}
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {"manifest": None, "recommendations": None, "rag_answer": output.strip()}
    except Exception as e:
        return {"error": f"Agent exception: {e}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant_key", required=True)
    parser.add_argument("--question", default="", help="Optional follow-up question")
    args = parser.parse_args()
    result = run_master(args.applicant_key, args.question)
    print(json.dumps(result, indent=2))
