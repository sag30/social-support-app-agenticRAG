import os
import pandas as pd
import streamlit as st
from datetime import datetime
from agent_orchestrator import run_master, rag_agent

os.makedirs("data/raw", exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOG_PATH = "logs/predictions.csv"
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=[
        "timestamp", "applicant_key",
        "upskilling_grant", "stipend", "counseling_voucher", "eligible"
    ]).to_csv(LOG_PATH, index=False)

st.set_page_config(page_title="Social Support Application", layout="centered")
st.title("Social Support Application Workflow Automation")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "manifest" not in st.session_state:
    st.session_state["manifest"] = None
if "result" not in st.session_state:
    st.session_state["result"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False

# ----------- 1. MAIN FORM -----------
st.header("Submit New Application")
with st.form("app_form"):
    applicant_key = st.text_input("Applicant Key (e.g. 'ahmad')", "")
    income         = st.number_input("Annual Income (AED)", min_value=0.0, value=0.0)
    employment_yrs = st.number_input("Years of Employment", min_value=0, value=0)
    family_size    = st.number_input("Family Size", min_value=1, value=1)
    
    bs_file  = st.file_uploader("Bank Statement (PDF)",     type=["pdf"])
    id_file  = st.file_uploader("Emirates ID (JPG/PNG)",    type=["jpg", "png", "jpeg"])
    cv_file  = st.file_uploader("Resume (DOCX)",            type=["docx"])
    al_file  = st.file_uploader("Assets/Liabilities (XLSX)",type=["xlsx"])
    cr_file  = st.file_uploader("Credit Report (PDF)",      type=["pdf"])
    submitted = st.form_submit_button("Submit & Get Recommendations")

if submitted:
    st.session_state["messages"] = []
    st.session_state["manifest"] = None
    st.session_state["result"] = None
    st.session_state["processing"] = True
    st.session_state["last_applicant_key"] = applicant_key
    # Save uploaded files to disk
    inputs = {
        "bank_statement":     bs_file,
        "EmiratesID":         id_file,
        "sample_resume":      cv_file,
        "assets_liabilities": al_file,
        "credit_report":      cr_file
    }
    missing_files = [k for k, v in inputs.items() if v is None]
    for name, uploaded in inputs.items():
        if uploaded is None:
            st.warning(f"{name} not uploaded; pipeline may skip it.")
            continue
        ext = os.path.splitext(uploaded.name)[1]
        path = f"data/raw/{name}_{applicant_key}{ext}"
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
    st.info("Processing your files. This may take a few seconds...")

    try:
        result = run_master(applicant_key, question=None)
        print("Shruti",result)
        st.session_state["result"] = result
        st.session_state["manifest"] = result.get("manifest", {})
        st.session_state["processing"] = False
        st.session_state["agent_result"] = result 
    except Exception as e:
        st.error(f"Agent pipeline failed: {e}")
        st.session_state["processing"] = False
        st.stop()

# ----------- DISPLAY LOGIC: ONLY SHOW AFTER SUBMISSION -----------
if st.session_state["processing"]:
    st.info("Processing your files. Please wait...")

if "agent_result" in st.session_state:
    result = st.session_state["agent_result"]
    manifest = result.get("manifest", {})
    st.subheader("Files Processed (from ETL Manifest)")
    files = manifest.get("files", [])
    if files:
        for f in files:
            st.write(f"- {f}")
    else:
        st.info("No files found or processed.")

if st.session_state.get("result"):
    st.subheader("Eligibility")
    recs = st.session_state["result"].get("recommendations", {})
    eligible = st.session_state["result"].get("eligible", None)
    if not recs:
        st.info("Not eligible for any programs at this time.")
    else:
        if eligible:
            st.success("Eligible for support!")
        else:
            st.info("Not eligible at this time.")
        st.write(pd.DataFrame(recs.items(), columns=["Program", "Score"]))

    # Log for auditability
    pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "applicant_key": applicant_key,
        "upskilling_grant": recs.get("Upskilling Grant", 0),
        "stipend":         recs.get("Stipend", 0),
        "counseling_voucher": recs.get("Career Counseling", 0),
        "eligible":        eligible
    }]).to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

# 2. Show chat at all times (but "Ask about Your Documents" is only enabled if applicant_key/result exists)
st.divider()
st.header("Chat about Your Documents")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# Only enable chat if an applicant_key is available
if applicant_key:
    chat_q = st.text_input("Ask a question about your application documents:", key="doc_chat")
    if st.button("Send", key="doc_chat_send"):
        try:
            # qa_result = run_master(applicant_key, question=chat_q)
            # rag_answer = "No answer."
            # if isinstance(result, dict):
            #     if "rag_answer" in result:
            #         rag_answer = result["rag_answer"]
            #     elif "result" in result:
            #         rag_answer = result["result"]
            #     elif isinstance(list(result.values())[0], str):
            #         rag_answer = list(result.values())[0]
            # st.session_state["messages"].append(("user", chat_q))
            # st.session_state["messages"].append(("agent", rag_answer))

        
            # Direct RAG call instead of agent, for full control
            rag_res = rag_agent(chat_q, applicant_key)
            rag_answer = rag_res.get("rag_answer", "No answer.")
            st.session_state["messages"].append(("user", chat_q))
            st.session_state["messages"].append(("agent", rag_answer))
            # Log for auditability
            pd.DataFrame([{
                "timestamp": datetime.now().isoformat(),
                "applicant_key": applicant_key,
                "question": chat_q,
                "rag_answer": rag_answer
            }]).to_csv("logs/chat_qa_audit.csv", mode="a", header=not os.path.exists("logs/chat_qa_audit.csv"), index=False)
        except Exception as e:
            st.session_state["messages"].append(("agent", f"Error: {e}"))
    for role, msg in st.session_state["messages"]:
        st.markdown(f"**{'You' if role == 'user' else 'Agent'}:** {msg}")
else:
    st.info("Please submit the application form above first.")