# app.py
# Consolidated script for MediCronus POC - Revised Button Logic & Prettier UI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import uuid
from pathlib import Path
import nest_asyncio

# --- Essential Imports ---
from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import asyncio # For nest_asyncio fallback

# Apply nest_asyncio
try:
    nest_asyncio.apply()
except RuntimeError:
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
             nest_asyncio.apply() # Try again if loop exists but isn't running
        else:
             st.warning("Asyncio loop already running. Nest_asyncio might not work as expected.")
    except RuntimeError: # If get_event_loop itself fails
         st.warning("Could not get or apply nest_asyncio. Async operations might behave unexpectedly.")
except Exception as e:
     st.warning(f"Could not apply nest_asyncio: {e}. Async operations might behave unexpectedly.")

# --- Environment Variables & API Key Setup ---
# (Keep this section as before - using os.getenv and st.secrets)
SERPER_API_KEY = os.getenv("SERPER_DEV_TOOL", st.secrets.get("SERPER_DEV_TOOL"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY"))
NEO4J_URI = os.getenv("NEO4J_URI", st.secrets.get("NEO4J_URI"))
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", st.secrets.get("NEO4J_USERNAME"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", st.secrets.get("NEO4J_PASSWORD"))

# Set environment variables for libraries
os.environ['SERPER_DEV_TOOL'] = SERPER_API_KEY if SERPER_API_KEY else ""
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
os.environ['GROQ_API_KEY'] = GROQ_API_KEY if GROQ_API_KEY else ""
os.environ['NEO4J_URI'] = NEO4J_URI if NEO4J_URI else ""
os.environ['NEO4J_USERNAME'] = NEO4J_USERNAME if NEO4J_USERNAME else ""
os.environ['NEO4J_PASSWORD'] = NEO4J_PASSWORD if NEO4J_PASSWORD else ""

# --- Check for Missing API Keys ---
# (Keep this section as before)
missing_keys = []
if not SERPER_API_KEY: missing_keys.append("SERPER_DEV_TOOL")
if not OPENAI_API_KEY: missing_keys.append("OPENAI_API_KEY (for embeddings)")
if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY (for chat LLM)")
if not NEO4J_URI: missing_keys.append("NEO4J_URI")
if not NEO4J_USERNAME: missing_keys.append("NEO4J_USERNAME")
if not NEO4J_PASSWORD: missing_keys.append("NEO4J_PASSWORD")

if missing_keys:
    st.error(f"🚨 Missing required secrets: {', '.join(missing_keys)}")
    st.warning("Please ensure all required API keys and credentials are set in your environment or Streamlit secrets.")
    st.stop()

# --- Crew Definitions ---
# (Keep Medic_Bot and Doctor_Bot class definitions exactly as before)
@CrewBase
class Medic_Bot():
    """Analyzes health reports."""
    @agent
    def report_extractor_agent(self) -> Agent:
        # ... agent definition ...
        return Agent(role="Extractor...", goal="...", backstory="...", memory=True, verbose=True, allow_delegation=False)
    @agent
    def report_explanation_agent(self) -> Agent:
        # ... agent definition ...
         return Agent(role="Explainer...", goal="...", backstory="...", memory=True, verbose=True, allow_delegation=False)
    @agent
    def abnormalities_agent(self) -> Agent:
        # ... agent definition ...
        return Agent(role="Abnormality Analyst...", goal="...", backstory="...", memory=True, verbose=True, allow_delegation=False)
    @task
    def report_extractor_task(self) -> Task:
        # ... task definition ...
         return Task(description="Extract from '{report}'...", expected_output="...", agent=self.report_extractor_agent())
    @task
    def report_explanation_task(self) -> Task:
        # ... task definition ...
        return Task(description="Analyze structured report...", expected_output="...", agent=self.report_explanation_agent(), context=[self.report_extractor_task()])
    @task
    def abnormalities_task(self) -> Task:
        # ... task definition ...
        return Task(description="Focus on abnormalities...", expected_output="...", context=[self.report_explanation_task()], agent=self.abnormalities_agent())
    @crew
    def crew(self) -> Crew:
        # ... crew definition ...
        return Crew(agents=[...], tasks=[...], process=Process.sequential, memory=True, verbose=2)

@CrewBase
class Doctor_Bot():
    """Finds doctors."""
    @agent
    def doctor_finder_agent(self) -> Agent:
        # ... agent definition ...
        return Agent(role="Finder...", goal="...", backstory="...", tools=[SerperDevTool()], verbose=True, memory=False, allow_delegation=False)
    @task
    def doctor_finder_task(self) -> Task:
        # ... task definition ...
        return Task(description="Search for doctors for '{abnormalities}' near '{user_city}'...", expected_output="...", agent=self.doctor_finder_agent())
    @crew
    def crew(self) -> Crew:
        # ... crew definition ...
        return Crew(agents=[...], tasks=[...], process=Process.sequential, verbose=2)


# --- Neo4j & Chatbot Setup Functions ---
# (Keep these functions as before: get_embeddings_model, get_llm,
#  setup_vector_store_and_retriever, create_chat_agent)
def get_embeddings_model():
    # ... function definition ...
    if not OPENAI_API_KEY: st.error("OpenAI Key missing"); return None
    try: return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e: st.error(f"Embeddings Error: {e}"); st.exception(e); return None

def get_llm():
    # ... function definition ...
    if not GROQ_API_KEY: st.error("Groq Key missing"); return None
    try: return ChatGroq(temperature=0.1, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    except Exception as e: st.error(f"LLM Error: {e}"); st.exception(e); return None

def setup_vector_store_and_retriever(docs_to_index: list, _embeddings):
     # ... function definition ...
    if not docs_to_index or not _embeddings: st.warning("No docs/embeddings for vector store"); return None, None
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]): st.error("Neo4j creds missing"); return None, None
    st.write("Debug: Indexing...")
    try:
        # Optional: Clear existing data
        try:
            st.write("Debug: Clearing Neo4j index...")
            db_clear = Neo4jVector.from_existing_index(_embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, index_name="vector", database="neo4j")
            db_clear.delete(delete_all=True)
            st.info("Cleared Neo4j index.")
        except Exception: st.warning("Could not clear Neo4j index (might be empty).")
        # Create
        db = Neo4jVector.from_documents(docs_to_index, _embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="neo4j", index_name="vector")
        st.success("✨ Knowledge Base Updated!")
        retriever = db.as_retriever()
        tool = create_retriever_tool(retriever, "Patient_Report_Analysis_Tool", "Searches patient's analyzed report...")
        return tool, db
    except Exception as e: st.error(f"Vector Store Error: {e}"); st.exception(e); return None, None

def create_chat_agent(_llm, _tool):
     # ... function definition ...
    if _tool is None or _llm is None: st.error("LLM/Tool missing for agent"); return None
    prompt = ChatPromptTemplate.from_messages([...]) # Keep prompt definition
    try:
        st.write("Debug: Creating chat agent...")
        agent = create_openai_tools_agent(llm=_llm, tools=[_tool], prompt=prompt)
        agent_exec = AgentExecutor(agent=agent, tools=[_tool], verbose=True, handle_parsing_errors=True, max_iterations=5)
        st.success("🤖 MediGuide Agent Ready!")
        return agent_exec
    except Exception as e: st.error(f"Chat Agent Error: {e}"); st.exception(e); return None


# --- File Handling & Text Processing ---
# (Keep these functions as before: read_file_content, split_text_for_indexing)
def read_file_content(file_path):
    # ... function definition ...
    try: return Path(file_path).read_text(encoding='utf-8')
    except Exception: return None

def split_text_for_indexing(text_content: str):
    # ... function definition ...
    if not text_content: return []
    rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    docs = rcts.split_text(text_content)
    return [Document(page_content=chunk) for chunk in docs]


# --- Streamlit App Main Logic ---
st.set_page_config(page_title="MediCronus POC", layout="wide", initial_sidebar_state="expanded")
st.title("⚕️ MediCronus - Report Analysis & Guidance")
st.markdown("Upload your PDF medical report for AI-powered analysis, optional doctor recommendations, and interactive chat.")
st.divider()

# Initialize Session State Variables more robustly
default_state = {
    'current_pdf_name': None, 'analysis_results': {}, 'doctor_recommendations': None,
    'analysis_triggered': False, 'doctors_triggered': False, 'kb_triggered': False,
    'analysis_complete': False, 'doctors_complete': False, 'kb_ready': False,
    'chat_agent_executor': None, 'chat_messages': [], 'uploaded_file_path': None,
    'user_wants_doctors': None, 'user_city': ""
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sidebar ---
with st.sidebar:
    st.header("📋 Controls")
    uploaded_file = st.file_uploader("1. Upload PDF Report", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        # Handle new file upload
        if st.session_state.current_pdf_name != uploaded_file.name:
            st.session_state.current_pdf_name = uploaded_file.name
            st.info(f"📄 File '{uploaded_file.name}' selected.")
            temp_dir = Path("./temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            temp_pdf_path = temp_dir / unique_filename
            try:
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = str(temp_pdf_path)
                st.success("✅ File ready for analysis.")
                # Reset state for new file
                for key, value in default_state.items(): # Reset all flags/results
                    st.session_state[key] = value
                st.session_state.current_pdf_name = uploaded_file.name # Keep name
                st.session_state.uploaded_file_path = str(temp_pdf_path) # Keep path
            except Exception as e:
                st.error(f"Error saving file: {e}")
                st.exception(e)
                st.session_state.uploaded_file_path = None
                st.session_state.current_pdf_name = None

    # Analyze Button - Active only if file is ready and analysis not done
    analyze_ready = bool(st.session_state.uploaded_file_path) and not st.session_state.analysis_complete
    if st.button("🔬 Analyze Report", key="analyze_button", disabled=not analyze_ready):
        # <<< --- BUTTON CLICK REGISTER CHECK --- >>>
        st.error("!!! Analyze Button Click Registered !!!")
        # <<< ------------------------------------ >>>
        st.session_state.analysis_triggered = True
        st.session_state.analysis_complete = False # Ensure reset
        st.session_state.doctors_complete = False
        st.session_state.kb_ready = False
        st.session_state.chat_messages = [] # Clear chat on new analysis
        st.rerun() # Rerun to show spinner

    st.divider()
    st.header("📊 Status")
    # Display Status Indicators
    if st.session_state.analysis_complete: st.success("✅ Analysis Complete")
    elif st.session_state.analysis_triggered: st.warning("⏳ Analysis Running...")
    else: st.info("⚪ Analysis Pending")

    if st.session_state.analysis_complete:
        doc_status = "⚪ N/A"
        if st.session_state.user_wants_doctors == 'Yes':
            if st.session_state.doctors_complete: doc_status = "✅ Doctors Searched"
            elif st.session_state.doctors_triggered: doc_status = "⏳ Searching Doctors..."
            else: doc_status = "⚪ Doctor Search Pending"
        elif st.session_state.user_wants_doctors == 'No':
            doc_status = "⚪ Doctors Skipped"
        st.info(doc_status)

        kb_status = "✅ MediGuide Ready" if st.session_state.kb_ready else "⏳ MediGuide Not Ready"
        if st.session_state.kb_triggered and not st.session_state.kb_ready: kb_status = "⏳ Building Knowledge..."
        st.info(kb_status)


# --- Main Area ---
col_results, col_chat = st.columns([3, 2])

with col_results:
    st.header("📈 Analysis & Recommendations")

    # --- Execute Analysis ---
    if st.session_state.analysis_triggered and not st.session_state.analysis_complete:
        with st.spinner("🔬 Processing report with Medic Bot Crew... Please wait."):
            try:
                st.write("Debug: Starting Medic Bot execution...")
                pdf_path = st.session_state.uploaded_file_path
                if not pdf_path or not Path(pdf_path).exists():
                     raise FileNotFoundError(f"PDF file lost or not found at {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                report_docs = loader.load()
                report_content = "\n\n".join([doc.page_content for doc in report_docs])
                if not report_content.strip(): raise ValueError("Empty PDF content")

                st.write("Debug: Initializing Medic Bot...")
                medic_crew = Medic_Bot().crew()
                inputs = {"report": report_content}
                st.write("Debug: Inputs ready for Medic Bot.")

                # <<< --- MEDIC BOT KICKOFF --- >>>
                st.write("Debug: Kicking off Medic Bot...")
                result = medic_crew.kickoff(inputs=inputs)
                st.write("Debug: Medic Bot finished.")
                # <<< ------------------------- >>>

                # Process results (same logic as before)
                st.session_state.analysis_results = {}
                if hasattr(result, 'tasks_output') and result.tasks_output:
                    st.write("Debug: Processing tasks_output from crew result.")
                    # ... (rest of the result processing logic) ...
                    if len(result.tasks_output) > 0 and hasattr(result.tasks_output[0], 'raw_output'): st.session_state.analysis_results["structured_report"] = result.tasks_output[0].raw_output
                    if len(result.tasks_output) > 1 and hasattr(result.tasks_output[1], 'raw_output'): st.session_state.analysis_results["explained_report"] = result.tasks_output[1].raw_output
                    if len(result.tasks_output) > 2 and hasattr(result.tasks_output[2], 'raw_output'): st.session_state.analysis_results["abnormalities_report"] = result.tasks_output[2].raw_output
                    else: st.session_state.analysis_results["abnormalities_report"] = result.tasks_output[-1].raw_output # Fallback
                elif hasattr(result, 'raw'): st.session_state.analysis_results["abnormalities_report"] = result.raw # Fallback
                else: st.warning("Could not parse crew results.")

                st.session_state.analysis_complete = True # Mark as complete

            except Exception as e:
                st.error(f"Analysis Failed: {e}")
                st.exception(e)
                st.session_state.analysis_complete = False # Ensure it's false on error
            finally:
                st.session_state.analysis_triggered = False # Reset trigger
                st.rerun() # Update UI


    # --- Display Analysis Results (if complete) ---
    if st.session_state.analysis_complete:
        tab1, tab2, tab3 = st.tabs(["📄 Structured Report", "🩺 Explained Report", "❗ Abnormalities Summary"])
        with tab1: st.markdown(st.session_state.analysis_results.get("structured_report", "*Not available.*"))
        with tab2: st.markdown(st.session_state.analysis_results.get("explained_report", "*Not available.*"))
        with tab3: st.markdown(st.session_state.analysis_results.get("abnormalities_report", "*Not available.*"))
        st.divider()

        # --- Doctor Recommendation Flow ---
        abnormalities_available = bool(st.session_state.analysis_results.get("abnormalities_report"))
        if not abnormalities_available:
            st.warning("Cannot search for doctors without abnormalities report.")
        elif st.session_state.user_wants_doctors is None:
             st.subheader("🧑‍⚕️ Find Specialists?")
             st.radio("Search for doctors based on abnormalities?", ("Yes", "No"),
                      key="want_doctors_radio", index=None, horizontal=True,
                      on_change=lambda: setattr(st.session_state, 'user_wants_doctors', st.session_state.want_doctors_radio))

        # --- Execute Doctor Search ---
        if st.session_state.user_wants_doctors == "Yes" and abnormalities_available and not st.session_state.doctors_complete:
            if not st.session_state.doctors_triggered: # Show only if not already running/done
                st.subheader("📍 Enter Location")
                city = st.text_input("Your City:", key="city_input", value=st.session_state.user_city)
                st.session_state.user_city = city
                if st.button("🔍 Find Doctors", key="find_doctors_button", disabled=not city):
                    # <<< --- BUTTON CLICK REGISTER CHECK --- >>>
                    st.error("!!! Find Doctors Button Click Registered !!!")
                    # <<< ------------------------------------ >>>
                    st.session_state.doctors_triggered = True
                    st.session_state.doctors_complete = False # Reset
                    st.rerun()

            # Actual execution block for doctor search
            if st.session_state.doctors_triggered and not st.session_state.doctors_complete:
                with st.spinner("🧑‍⚕️ Searching for doctors with Doctor Bot Crew..."):
                    try:
                        st.write("Debug: Starting Doctor Bot execution...")
                        doctor_crew = Doctor_Bot().crew()
                        abnormalities_summary = st.session_state.analysis_results.get("abnormalities_report", "General Health Checkup")
                        current_city = st.session_state.user_city
                        inputs = {"user_city": current_city, "abnormalities": abnormalities_summary}
                        st.write(f"Debug: Inputs ready for Doctor Bot (City: {current_city})")

                        # <<< --- DOCTOR BOT KICKOFF --- >>>
                        st.write("Debug: Kicking off Doctor Bot...")
                        doctor_result = doctor_crew.kickoff(inputs=inputs)
                        st.write("Debug: Doctor Bot finished.")
                        # <<< ------------------------ >>>

                        st.session_state.doctor_recommendations = doctor_result.raw if hasattr(doctor_result, 'raw') else "*No recommendations found.*"
                        st.session_state.doctors_complete = True

                    except Exception as e:
                        st.error(f"Doctor Search Failed: {e}")
                        st.exception(e)
                        st.session_state.doctors_complete = False
                    finally:
                        st.session_state.doctors_triggered = False # Reset trigger
                        st.rerun()

        # Display doctor results if search was completed
        if st.session_state.doctors_complete:
            st.subheader("🧑‍⚕️ Doctor Recommendations")
            with st.expander("View Found Specialists", expanded=True):
                 st.markdown(st.session_state.get('doctor_recommendations', "*No details available.*"))

        # Acknowledge if user chose No
        if st.session_state.user_wants_doctors == "No":
            st.info("⚪ Skipping doctor recommendations.")


        # --- Knowledge Base Building (Trigger Automatically) ---
        ready_for_kb = st.session_state.analysis_complete and \
                       (st.session_state.user_wants_doctors == 'No' or st.session_state.doctors_complete)

        if ready_for_kb and not st.session_state.kb_ready and not st.session_state.kb_triggered:
            st.write("Debug: Triggering Knowledge Base build.") # Debug
            st.session_state.kb_triggered = True
            st.rerun()

        if st.session_state.kb_triggered and not st.session_state.kb_ready:
            with st.spinner("🧠 Preparing MediGuide Assistant... Building Knowledge Base"):
                try:
                    st.write("Debug: Gathering content for KB...")
                    # (Keep logic for gathering content from analysis_results and doctor_recommendations)
                    content_to_index = [...] # Gather content list
                    full_text = "\n\n---\n\n".join(content_to_index)
                    if not full_text.strip(): raise ValueError("No content for KB")

                    st.write("Debug: Splitting documents for KB...")
                    doc_chunks = split_text_for_indexing(full_text)
                    if not doc_chunks: raise ValueError("No doc chunks for KB")

                    st.write("Debug: Setting up vector store for KB...")
                    embeddings = get_embeddings_model()
                    if embeddings:
                         retriever_tool, _ = setup_vector_store_and_retriever(doc_chunks, embeddings)
                         if retriever_tool:
                            st.write("Debug: Initializing chat agent for KB...")
                            llm = get_llm()
                            if llm:
                                agent_executor = create_chat_agent(llm, retriever_tool)
                                if agent_executor:
                                    st.session_state.chat_agent_executor = agent_executor
                                    st.session_state.kb_ready = True
                                    st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! I'm MediGuide. Ask me questions about your analyzed report."}]
                                else: st.error("Failed to create chat agent executor.")
                            else: st.error("Failed to initialize LLM.")
                         else: st.error("Failed to setup retriever tool.")
                    else: st.error("Failed to initialize embeddings model.")

                except Exception as e:
                    st.error(f"Knowledge Base Setup Failed: {e}")
                    st.exception(e)
                    st.session_state.kb_ready = False
                finally:
                     st.session_state.kb_triggered = False # Reset trigger
                     st.rerun()


    # Initial Prompt if no analysis done yet
    elif not st.session_state.analysis_triggered and not st.session_state.analysis_complete :
         st.info("⬅️ Please upload a PDF report using the sidebar to start.")


with col_chat:
    st.header("💬 Chat with MediGuide")

    if not st.session_state.kb_ready:
        st.info("🤖 MediGuide will become available here once the report analysis and knowledge base preparation are complete.")
        if st.session_state.kb_triggered: st.warning("🧠 Building knowledge...")

    else:
        # Chat Interface
        # Display existing messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get new user input
        if prompt := st.chat_input("Ask about your analyzed report..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            if st.session_state.chat_agent_executor:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    try:
                        # (Keep logic for preparing chat history and invoking agent)
                        langchain_chat_history = [...] # Prepare history
                        st.write("Debug: Invoking chat agent...")
                        response = st.session_state.chat_agent_executor.invoke({"input": prompt, "chat_history": langchain_chat_history})
                        response_content = response.get("output", "*Error generating response.*")
                        st.write("Debug: Chat agent invocation complete.")
                    except Exception as e:
                        st.error(f"Chat Error: {e}"); st.exception(e)
                        response_content = "*Sorry, an error occurred.*"
                    message_placeholder.markdown(response_content)
                st.session_state.chat_messages.append({"role": "assistant", "content": response_content})
            else:
                st.error("Chat agent is not ready.")

# --- Footer ---
st.divider()
st.caption("MediCronus POC | Powered by CrewAI, LangChain, Groq, Neo4j, Streamlit")
