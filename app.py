# app.py
# Consolidated script for MediCronus POC - with added debugging prints
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import uuid # To create unique filenames for uploads
from pathlib import Path
import nest_asyncio # For handling asyncio event loops

# --- Essential Imports ---
from crewai import Agent, Task, Crew, CrewBase
from crewai.process import Process
from crewai.project import agent, task, crew # For decorator usage
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
# Optional: If your agent needs specific message types
# from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document # Added for split_text_for_indexing

# Apply nest_asyncio for environments like Streamlit that might have an event loop
try:
    nest_asyncio.apply()
except RuntimeError:
    # Handle cases where it might already be applied or not needed
    # Example policy setting (might vary based on OS/environment)
    import asyncio
    try:
        # Attempt setting a policy if possible (e.g., for Windows)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        nest_asyncio.apply() # Try applying again after setting policy
    except Exception: # Broad except as policy setting might not be available/needed
        st.warning(f"Could not apply nest_asyncio; proceeding. Async operations might behave unexpectedly.")
except Exception as e:
     st.warning(f"Could not apply nest_asyncio: {e}. Async operations might behave unexpectedly.")


# --- Environment Variables & API Key Setup ---
# For local development, ensure these are set in your environment
# For Streamlit Sharing/Community Cloud, use st.secrets
SERPER_API_KEY = os.getenv("SERPER_DEV_TOOL", st.secrets.get("SERPER_DEV_TOOL"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY"))
NEO4J_URI = os.getenv("NEO4J_URI", st.secrets.get("NEO4J_URI"))
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", st.secrets.get("NEO4J_USERNAME"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", st.secrets.get("NEO4J_PASSWORD"))

# Set environment variables for libraries that expect them
# Important: Ensure these are set BEFORE initializing tools/models that need them
os.environ['SERPER_DEV_TOOL'] = SERPER_API_KEY if SERPER_API_KEY else ""
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
os.environ['GROQ_API_KEY'] = GROQ_API_KEY if GROQ_API_KEY else ""
# Setting Neo4j env vars here is optional if passed directly during Neo4jVector init
os.environ['NEO4J_URI'] = NEO4J_URI if NEO4J_URI else ""
os.environ['NEO4J_USERNAME'] = NEO4J_USERNAME if NEO4J_USERNAME else ""
os.environ['NEO4J_PASSWORD'] = NEO4J_PASSWORD if NEO4J_PASSWORD else ""

# --- Check for Missing API Keys ---
missing_keys = []
# Add mandatory keys here
if not SERPER_API_KEY: missing_keys.append("SERPER_DEV_TOOL")
if not OPENAI_API_KEY: missing_keys.append("OPENAI_API_KEY (for embeddings)")
if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY (for chat LLM)")
if not NEO4J_URI: missing_keys.append("NEO4J_URI")
if not NEO4J_USERNAME: missing_keys.append("NEO4J_USERNAME")
if not NEO4J_PASSWORD: missing_keys.append("NEO4J_PASSWORD")

if missing_keys:
    st.error(f"Missing required environment variables or secrets: {', '.join(missing_keys)}")
    st.warning("Please ensure all required API keys and credentials are set.")
    st.stop() # Halt execution if critical keys are missing


# --- Crew Definitions (Included directly in app.py) ---

@CrewBase
class Medic_Bot():
    """The Medic Bot analyzes the health report and generates comprehensive and useful insights"""
    # --- Agent Definitions ---
    @agent
    def report_extractor_agent(self) -> Agent:
        return Agent(
            role="Extract, organize, and structure report contents into a clear and well-formatted document.",
            goal="Produce a logically structured and comprehensive report that enhances readability while preserving all information.",
            backstory="You are an expert in report structuring...", # Keep full backstory
            memory=True, verbose=True, allow_delegation=False
        )

    @agent
    def report_explanation_agent(self) -> Agent:
         return Agent(
            role="Analyze the report received from the report_extractor_agent, evaluate all values, and determine which fall within the normal range and which deviate.",
            goal="Generate a well-structured and comprehensive report that classifies values as normal or abnormal, explains their significance, and identifies potential health implications based *only* on the provided report data.",
            backstory="You are an experienced pathologist...", # Keep full backstory
            memory=True, verbose=True, allow_delegation=False
        )

    @agent
    def abnormalities_agent(self) -> Agent:
        return Agent(
            role="Analyze the report received from the report_explanation_agent, identify abnormal values, and focus on those. "
            "Provide potential actionable recommendations for lifestyle changes or diet based *only* on the identified abnormalities in the report. "
            "Assess the potential severity of abnormalities based on deviation from normal ranges and determine if medical consultation is generally advisable for such findings.",
            goal="Generate a structured and detailed report that highlights abnormal values, explains their significance based on the report context, "
            "and suggests general, non-prescriptive recommendations for discussion with a healthcare professional. Flag findings that typically warrant medical attention.",
            backstory="You are a highly experienced medical analyst...", # Keep full backstory
            memory=True, verbose=True, allow_delegation=False
        )

    # --- Task Definitions ---
    @task
    def report_extractor_task(self) -> Task:
        return Task(
            description="Extract and organize the contents of the provided medical report: '{report}'. "
            "Ensure logical flow, clarity, and readability by using appropriate headings, subheadings, and formatting. Preserve all numerical values and test names accurately.",
            expected_output="A complete and professionally structured report in Markdown format (.md) that maintains all essential information while improving organization "
            "and presentation. The final document should be clear, concise, and easy to navigate.",
            agent=self.report_extractor_agent()
        )

    @task
    def report_explanation_task(self) -> Task:
        return Task(
            description="Analyze the structured report provided by the report_extractor_agent. "
            "Evaluate all numerical and categorical values against standard reference ranges (if provided in the report, otherwise use general knowledge cautiously). Classify them as normal or abnormal. "
            "Provide a detailed explanation for each value (especially abnormal ones), including its general function or significance in the body and potential implications of deviation. "
            "Assign a potential severity indication (e.g., Mild, Moderate, Significant) based on the degree of deviation for abnormal values. "
            "Ensure the report is structured for clarity, using headings, subheadings, and bullet points.",
            expected_output="A well-organized, detailed medical explanation report in Markdown format (.md) that: \n"
            "- Clearly lists each parameter analyzed.\n"
            "- Classifies each value as normal or abnormal.\n"
            "- Assigns a potential severity indication to abnormalities.\n"
            "- Provides meaningful explanations for values, focusing on abnormalities, including possible general implications.\n"
            "- Is structured with clear headings, subheadings, and bullet points for readability.\n",
            agent=self.report_explanation_agent(),
            context=[self.report_extractor_task()] # Ensure context passing works
        )

    @task
    def abnormalities_task(self) -> Task:
        return Task(
             description="Analyze the explained report from the report_explanation_agent, focusing *exclusively* on the identified abnormal values. "
             "For each abnormal metric: summarize its potential significance, explain *why* it's considered abnormal based on the report context, "
             "and provide general, non-prescriptive recommendations for lifestyle or diet relevant to such findings (e.g., 'Consider discussing dietary fiber intake with your doctor' if cholesterol is high). "
             "Reiterate the potential severity. Indicate if the type of abnormality typically warrants medical consultation and suggest the type of specialist (e.g., Endocrinologist, Cardiologist) usually involved.",
             expected_output="A structured and detailed abnormalities summary report in Markdown format (.md) that:\n"
             "- Lists *only* abnormal values along with their potential medical significance.\n"
             "- Explains the reason for the abnormal classification based on the report.\n"
             "- Provides general, non-prescriptive recommendations for discussion with a doctor.\n"
             "- Includes the potential severity score assigned earlier.\n"
             "- Recommends consulting a healthcare professional and suggests relevant specialist types.\n"
             "- Is formatted with clear headings, subheadings, and bullet points.",
             context=[self.report_explanation_task()],
             agent=self.abnormalities_agent(),
        )

    # --- Crew Definition ---
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.report_extractor_agent(), self.report_explanation_agent(), self.abnormalities_agent()],
            tasks=[self.report_extractor_task(), self.report_explanation_task(), self.abnormalities_task()],
            process=Process.sequential, # Tasks run in order
            memory=True, # Enable memory for context passing between tasks
            verbose=2 # Or 1 for less detailed logs, 0 for silent
        )

@CrewBase
class Doctor_Bot():
    """This bot will find the best doctors for the user based on abnormalities and location."""
    # --- Agent Definition ---
    @agent
    def doctor_finder_agent(self) -> Agent:
        return Agent(
            role="Specialist Doctor Finder Agent",
            goal="Find highly-rated doctors specializing in the specific medical conditions or abnormalities provided, within the user's specified city or nearby areas. Provide detailed contact and practice information.",
            backstory="You are an AI-powered healthcare assistant equipped with advanced web search capabilities (Serper). "
            "Your purpose is to accurately identify relevant medical specialists (like Hematologists, Endocrinologists, Cardiologists, etc.) based on the patient's specific health report abnormalities and their location. "
            "You prioritize providing actionable information to help the user connect with potential doctors.",
            tools=[SerperDevTool()], # Use the search tool
            verbose=True,
            memory=False, # This agent likely doesn't need long-term memory for a single search
            allow_delegation=False
        )

    # --- Task Definition ---
    @task
    def doctor_finder_task(self) -> Task:
        return Task(
            description="Perform a web search to find top-rated medical specialists relevant to these abnormalities: '{abnormalities}'. \n"
            "The search must focus on doctors located in or very close to the city: '{user_city}'. \n"
            "For each relevant specialist found (aim for 2-5 if possible), gather the following details:\n"
            "1. Doctor's Full Name\n"
            "2. Medical Specialization (e.g., Endocrinologist, Hematologist)\n"
            "3. Name of the Clinic or Hospital\n"
            "4. Full Address of the practice\n"
            "5. Contact Phone Number\n"
            "6. Website or Profile URL (if available)\n"
            "7. Brief summary of patient reviews or ratings found (if available).",
            expected_output="A list of 2-5 recommended specialists formatted clearly in Markdown (.md). Each entry should include all requested details (Name, Specialization, Clinic/Hospital, Address, Phone, URL, Reviews Summary). If no relevant doctors are found, state that clearly.",
            agent=self.doctor_finder_agent(),
        )

    # --- Crew Definition ---
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.doctor_finder_agent()],
            tasks=[self.doctor_finder_task()],
            process=Process.sequential,
            verbose=2
        )


# --- Neo4j & Chatbot Setup Functions ---
# @st.cache_resource # Cache resource-intensive objects
def get_embeddings_model():
    """Initializes and returns the OpenAI embeddings model."""
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Cannot initialize embeddings model.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings: {e}")
        st.exception(e) # Show traceback
        return None

# @st.cache_resource
def get_llm():
    """Initializes and returns the Groq LLM."""
    if not GROQ_API_KEY:
        st.error("Groq API Key not found. Cannot initialize chat LLM.")
        return None
    try:
        # Recommended Llama3 model
        return ChatGroq(temperature=0.1, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")
        st.exception(e) # Show traceback
        return None

# @st.cache_resource # Cache the connection if credentials don't change often
def setup_vector_store_and_retriever(docs_to_index: list, _embeddings):
    """Loads documents, creates Neo4j vector store, and returns a retriever tool."""
    if not docs_to_index or not _embeddings:
        st.warning("No documents or embeddings model provided for vector store setup.")
        return None, None
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        st.error("Neo4j connection details missing. Cannot create vector store.")
        return None, None

    st.write(f"Debug: Preparing to index {len(docs_to_index)} document chunks...")
    try:
        # Clear existing data first (optional, consider implications)
        try:
            st.write("Debug: Attempting to connect to existing Neo4j index for clearing...")
            temp_db_to_clear = Neo4jVector.from_existing_index(
                embedding=_embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="vector", database="neo4j",
            )
            st.write("Debug: Deleting all nodes in 'vector' index...")
            temp_db_to_clear.delete(delete_all=True)
            st.info("Cleared existing data in Neo4j 'vector' index.")
        except Exception as e:
            st.warning(f"Could not clear Neo4j index (may be empty or first run): {e}")

        # Create the vector store
        st.write("Debug: Creating new Neo4j vector store from documents...")
        db = Neo4jVector.from_documents(
            documents=docs_to_index, embedding=_embeddings, url=NEO4J_URI,
            username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="neo4j", index_name="vector",
        )
        st.success("Neo4j Vector Store created/updated successfully!")
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever, "Patient_Report_Analysis_Tool",
            "Use this tool to search for information within the patient's analyzed medical report..." # Keep description
        )
        return tool, db
    except Exception as e:
        st.error(f"Failed to create/update Neo4j Vector Store: {e}")
        st.exception(e) # Show traceback
        return None, None

# @st.cache_resource # Cache the agent executor
def create_chat_agent(_llm, _tool):
    """Creates the Langchain AgentExecutor for the chatbot."""
    if _tool is None or _llm is None:
        st.error("LLM or Retriever Tool is missing. Cannot create chat agent.")
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are MediGuide, a helpful AI assistant designed to discuss the patient's *most recently analyzed* medical report..."), # Keep prompt
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    try:
        st.write("Debug: Creating OpenAI Tools agent...")
        agent = create_openai_tools_agent(llm=_llm, tools=[_tool], prompt=prompt)
        st.write("Debug: Creating Agent Executor...")
        agent_exec = AgentExecutor(
            agent=agent, tools=[_tool], verbose=True, handle_parsing_errors=True, max_iterations=5
        )
        st.success("Chat agent created successfully.")
        return agent_exec
    except Exception as e:
        st.error(f"Failed to create chat agent: {e}")
        st.exception(e) # Show traceback
        return None

# --- File Handling & Text Processing ---
def read_file_content(file_path):
    """Reads content from a file path."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return None

def split_text_for_indexing(text_content: str):
    """Splits text content into chunks suitable for embedding."""
    if not text_content:
        return []
    rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    docs = rcts.split_text(text_content)
    # Convert simple text chunks into Langchain Document objects for Neo4jVector
    return [Document(page_content=chunk) for chunk in docs]


# --- Streamlit App Main Logic ---
st.set_page_config(page_title="MediCronus POC", layout="wide")
st.title("⚕️ MediCronus - Report Analysis & Guidance")
st.subheader("Upload a PDF medical report for analysis and chat about the results.")

# Initialize Session State Variables (Crucial for Streamlit apps)
# Use more descriptive keys
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {} # Store structured results from crews
if 'doctor_recommendations' not in st.session_state:
    st.session_state.doctor_recommendations = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'doctors_running' not in st.session_state:
    st.session_state.doctors_running = False
if 'kb_building' not in st.session_state:
    st.session_state.kb_building = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'doctors_complete' not in st.session_state:
    st.session_state.doctors_complete = False # Tracks if doctor search was *attempted*
if 'kb_ready' not in st.session_state:
    st.session_state.kb_ready = False
if 'chat_agent_executor' not in st.session_state:
    st.session_state.chat_agent_executor = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [] # Store chat history: [{"role": "user/assistant", "content": "..."}]
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'user_wants_doctors' not in st.session_state:
    st.session_state.user_wants_doctors = None # 'Yes', 'No', or None (undecided)
if 'user_city' not in st.session_state:
    st.session_state.user_city = ""

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("1. Upload PDF Report", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        st.write(f"Uploaded: **{uploaded_file.name}**")
        # Save/update file path only if it's a new file
        if st.session_state.current_pdf_name != uploaded_file.name:
            st.session_state.current_pdf_name = uploaded_file.name
            temp_dir = Path("./temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            temp_pdf_path = temp_dir / unique_filename
            try:
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = str(temp_pdf_path)
                st.success("File ready for analysis.")
                # Reset status flags for the new file
                st.session_state.analysis_complete = False
                st.session_state.doctors_complete = False
                st.session_state.kb_ready = False
                st.session_state.chat_agent_executor = None
                st.session_state.chat_messages = []
                st.session_state.analysis_results = {}
                st.session_state.doctor_recommendations = None
                st.session_state.user_wants_doctors = None
                st.session_state.user_city = ""
                st.session_state.analysis_running = False # Ensure running flags are reset
                st.session_state.doctors_running = False
                st.session_state.kb_building = False
            except Exception as e:
                st.error(f"Error saving uploaded file: {e}")
                st.exception(e)
                st.session_state.uploaded_file_path = None
                st.session_state.current_pdf_name = None

    # --- Add Debug State Check RIGHT BEFORE the Button ---
    st.sidebar.divider()
    st.sidebar.markdown("--- **Debug Info** ---")
    st.sidebar.write(f"**State before Analyze Button:**")
    # Check if path exists and has a value before accessing properties
    path_value = st.session_state.get('uploaded_file_path', None)
    st.sidebar.write(f"  - uploaded_file_path: {'Set' if path_value else 'None'}")
    st.sidebar.write(f"  - analysis_complete: {st.session_state.get('analysis_complete', False)}")
    st.sidebar.write(f"  - analysis_running: {st.session_state.get('analysis_running', False)}")
    st.sidebar.markdown("---")
    # --- End Debug State Check ---


    # Button to trigger analysis - only active if a file path is set and analysis not complete/running
    if st.session_state.get('uploaded_file_path') and not st.session_state.get('analysis_complete', False):
        analyze_disabled = st.session_state.get('analysis_running', False)
        if st.button("2. Analyze Report", key="analyze_button", disabled=analyze_disabled):
            st.write("Debug: Analyze button clicked.") # Debug print
            st.session_state.analysis_running = True
            # Reset downstream states
            st.session_state.analysis_complete = False
            st.session_state.doctors_complete = False
            st.session_state.kb_ready = False
            st.session_state.chat_agent_executor = None
            st.session_state.chat_messages = []
            st.rerun() # Rerun immediately to show spinner and disable button

    # Execute analysis if flag is set (handles the rerun from button click)
    if st.session_state.get('analysis_running') and not st.session_state.get('analysis_complete'):
            with st.spinner("Analyzing report... This may take a few moments."):
                try:
                    st.write("Debug: Inside Analyze try block.")
                    pdf_path = st.session_state.uploaded_file_path
                    st.write(f"Debug: Loading PDF from {pdf_path}")
                    if not pdf_path or not Path(pdf_path).exists():
                         raise FileNotFoundError(f"PDF file not found at {pdf_path}")

                    loader = PyPDFLoader(pdf_path)
                    report_docs = loader.load()
                    report_content = "\n\n".join([doc.page_content for doc in report_docs])
                    st.write(f"Debug: PDF content length: {len(report_content)}")

                    if not report_content.strip():
                         st.error("Failed to extract text content from the PDF.")
                         raise ValueError("Empty report content after loading")

                    st.write("Debug: Initializing Medic Bot Crew...")
                    medic_crew = Medic_Bot().crew()
                    st.write("Debug: Medic Bot Crew Initialized.")

                    inputs = {"report": report_content}
                    st.write(f"Debug: Inputs for Medic Bot: {list(inputs.keys())}")

                    st.write("Debug: Kicking off Medic Bot Crew...")
                    result = medic_crew.kickoff(inputs=inputs)
                    st.write("Debug: Medic Bot Crew kickoff finished.")

                    # Process results carefully
                    st.session_state.analysis_results = {}
                    if hasattr(result, 'tasks_output') and result.tasks_output:
                        st.write("Debug: Processing tasks_output from crew result.")
                        if len(result.tasks_output) > 0 and hasattr(result.tasks_output[0], 'raw_output'):
                            st.session_state.analysis_results["structured_report"] = result.tasks_output[0].raw_output
                        if len(result.tasks_output) > 1 and hasattr(result.tasks_output[1], 'raw_output'):
                            st.session_state.analysis_results["explained_report"] = result.tasks_output[1].raw_output
                        if len(result.tasks_output) > 2 and hasattr(result.tasks_output[2], 'raw_output'):
                            st.session_state.analysis_results["abnormalities_report"] = result.tasks_output[2].raw_output
                        else: # Fallback for abnormalities if last task output isn't indexed as expected
                             st.session_state.analysis_results["abnormalities_report"] = result.tasks_output[-1].raw_output
                    elif hasattr(result, 'raw'): # Fallback if tasks_output structure is different
                         st.write("Debug: Using result.raw as fallback for abnormalities report.")
                         st.session_state.analysis_results["abnormalities_report"] = result.raw
                    else:
                        st.warning("Could not parse crew results effectively. Check crew output structure.")

                    st.session_state.analysis_complete = True
                    st.success("Report analysis complete!")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.exception(e) # Print full traceback in Streamlit app
                    st.session_state.analysis_complete = False # Ensure it's marked as not complete on error
                finally:
                    st.session_state.analysis_running = False # Mark as not running anymore
                    # Optionally cleanup temp file here
                    # if pdf_path and Path(pdf_path).exists():
                    #    try: Path(pdf_path).unlink() except OSError: pass
                    #    st.session_state.uploaded_file_path = None
                    st.rerun() # Rerun to update UI based on completion status

    # Display current status
    st.sidebar.divider()
    st.sidebar.write("**Status:**")
    status_analysis = "✅ Analysis Complete" if st.session_state.get('analysis_complete') else "⏳ Analysis Pending/Failed"
    st.sidebar.info(status_analysis)

    if st.session_state.get('analysis_complete'): # Only show doctor/KB status after analysis
        status_doctors = "⚪ Not Requested"
        if st.session_state.get('user_wants_doctors') == 'Yes':
            status_doctors = "✅ Doctors Searched" if st.session_state.get('doctors_complete') else "⏳ Doctor Search Pending/Failed"
        elif st.session_state.get('user_wants_doctors') == 'No':
            status_doctors = "⚪ Doctors Skipped"
        st.sidebar.info(status_doctors)

        status_kb = "✅ MediGuide Ready" if st.session_state.get('kb_ready') else "⏳ MediGuide Not Ready"
        st.sidebar.info(status_kb)


# --- Main Area Layout (Report Display & Chat) ---
col1, col2 = st.columns([3, 2]) # Give more space to report display

with col1:
    st.header("Analysis Results")
    if not st.session_state.get('analysis_complete'):
        if st.session_state.get('uploaded_file_path'):
             st.info("Report uploaded. Click 'Analyze Report' in the sidebar to begin.")
        elif st.session_state.get('analysis_running'):
             st.info("Analysis is in progress...") # Message while running
        else:
             st.info("Upload a PDF report using the sidebar to start the process.")
    else:
        # --- Display Analysis Results in Tabs ---
        tab1, tab2, tab3 = st.tabs(["📄 Structured Report", "🩺 Explained Report", "❗ Abnormalities Summary"])
        with tab1:
            report_md = st.session_state.analysis_results.get("structured_report", "*No structured report generated.*")
            st.markdown(report_md)
        with tab2:
            explained_md = st.session_state.analysis_results.get("explained_report", "*No explained report generated.*")
            st.markdown(explained_md)
        with tab3:
            abnormal_md = st.session_state.analysis_results.get("abnormalities_report", "*No abnormalities summary generated.*")
            st.markdown(abnormal_md)

        st.divider()

        # --- Doctor Recommendation Flow ---
        abnormalities_available = bool(st.session_state.analysis_results.get("abnormalities_report"))
        if not abnormalities_available:
            st.warning("Cannot search for doctors as abnormalities report was not generated or is empty.")
        elif st.session_state.user_wants_doctors is None: # Ask only once per analysis
             st.subheader("Find Specialists?")
             st.radio(
                 "Would you like to search for recommended doctors based on the abnormalities?",
                 ("Yes", "No"), key="want_doctors_radio", index=None,
                 on_change=lambda: setattr(st.session_state, 'user_wants_doctors', st.session_state.want_doctors_radio)
             )

        # Section for finding doctors
        if st.session_state.user_wants_doctors == "Yes" and abnormalities_available:
            if not st.session_state.get('doctors_complete', False):
                st.subheader("Enter Location")
                city = st.text_input("Your City:", key="city_input", value=st.session_state.user_city)
                st.session_state.user_city = city

                find_doctors_disabled = st.session_state.get('doctors_running', False) or not city
                if st.button("3. Find Doctors", key="find_doctors_button", disabled=find_doctors_disabled):
                    st.write("Debug: Find Doctors button clicked.") # Debug
                    st.session_state.doctors_running = True
                    st.session_state.doctors_complete = False # Reset before starting search
                    st.rerun() # Show spinner

            # Execute doctor search if flag is set
            if st.session_state.get('doctors_running') and not st.session_state.get('doctors_complete'):
                 with st.spinner("Searching for relevant doctors..."):
                        try:
                            st.write("Debug: Inside Find Doctors try block.") # Debug
                            st.write(f"Debug: Initializing Doctor Bot Crew...") # Debug
                            doctor_crew = Doctor_Bot().crew()
                            st.write("Debug: Doctor Bot Crew initialized.") # Debug

                            abnormalities_summary = st.session_state.analysis_results.get("abnormalities_report", "General Health Checkup")
                            current_city = st.session_state.user_city
                            inputs = {"user_city": current_city, "abnormalities": abnormalities_summary}
                            st.write(f"Debug: Inputs for Doctor Bot: user_city='{current_city}', abnormalities_len={len(inputs['abnormalities'])}") # Debug

                            st.write("Debug: Kicking off Doctor Bot Crew...") # Debug
                            doctor_result = doctor_crew.kickoff(inputs=inputs)
                            st.write("Debug: Doctor Bot Crew kickoff finished.") # Debug

                            st.session_state.doctor_recommendations = doctor_result.raw if hasattr(doctor_result, 'raw') else "*Could not retrieve doctor recommendations.*"
                            st.session_state.doctors_complete = True # Mark as complete (search attempted)
                            st.success("Doctor search finished.")

                        except Exception as e:
                            st.error(f"An error occurred during doctor search: {e}")
                            st.exception(e) # Print full traceback
                            st.session_state.doctors_complete = False # Mark as failed
                        finally:
                            st.session_state.doctors_running = False
                            st.rerun() # Update UI

            # Display doctor results if search was completed
            if st.session_state.get('doctors_complete'):
                st.subheader("Doctor Recommendations")
                if st.session_state.get('doctor_recommendations'):
                    with st.expander("View Found Specialists", expanded=True):
                        st.markdown(st.session_state.doctor_recommendations)
                else:
                    st.info("The search completed, but no specific doctor recommendations were generated.")

        # Acknowledge if user chose No
        if st.session_state.user_wants_doctors == "No":
            st.info("Skipping doctor recommendations.")


        # --- Knowledge Base Building (Trigger Automatically after Analysis/Doctor steps) ---
        ready_for_kb = st.session_state.get('analysis_complete') and \
                       (st.session_state.get('user_wants_doctors') == 'No' or st.session_state.get('doctors_complete'))

        if ready_for_kb and not st.session_state.get('kb_ready') and not st.session_state.get('kb_building'):
            st.write("Debug: Conditions met for KB building.") # Debug
            st.session_state.kb_building = True
            st.rerun() # Show spinner

        if st.session_state.get('kb_building'):
            with st.spinner("Preparing MediGuide Assistant... Building Knowledge Base"):
                try:
                    st.write("Debug: Gathering analyzed content for KB...") # Debug
                    content_to_index = []
                    if st.session_state.analysis_results.get("structured_report"):
                        content_to_index.append(st.session_state.analysis_results["structured_report"])
                    if st.session_state.analysis_results.get("explained_report"):
                        content_to_index.append(st.session_state.analysis_results["explained_report"])
                    if st.session_state.analysis_results.get("abnormalities_report"):
                        content_to_index.append(st.session_state.analysis_results["abnormalities_report"])
                    if st.session_state.get('doctor_recommendations'):
                        content_to_index.append(st.session_state.doctor_recommendations)

                    full_text = "\n\n---\n\n".join(content_to_index)

                    if not full_text.strip():
                        st.error("No content available to build knowledge base.")
                        raise ValueError("Empty content for KB")

                    st.write("Debug: Splitting documents for KB...") # Debug
                    doc_chunks = split_text_for_indexing(full_text)

                    if not doc_chunks:
                         st.error("Failed to split documents for indexing.")
                         raise ValueError("No document chunks created")

                    st.write("Debug: Setting up vector store for KB...") # Debug
                    embeddings = get_embeddings_model()
                    if embeddings:
                         retriever_tool, vector_db = setup_vector_store_and_retriever(doc_chunks, embeddings)
                         if retriever_tool:
                            st.write("Debug: Initializing chat agent for KB...") # Debug
                            llm = get_llm()
                            if llm:
                                agent_executor = create_chat_agent(llm, retriever_tool)
                                if agent_executor:
                                    st.session_state.chat_agent_executor = agent_executor
                                    st.session_state.kb_ready = True
                                    st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! I'm MediGuide. Ask me questions about your analyzed report."}] # Start with greeting
                                    st.success("MediGuide Assistant is ready!")
                                else: st.error("Failed to create chat agent executor.")
                            else: st.error("Failed to initialize LLM for chat agent.")
                         else: st.error("Failed to setup retriever tool for knowledge base.")
                    else: st.error("Failed to initialize embeddings model.")

                except Exception as e:
                    st.error(f"Failed to build knowledge base or setup chatbot: {e}")
                    st.exception(e)
                    st.session_state.kb_ready = False
                finally:
                     st.session_state.kb_building = False
                     st.rerun() # Update UI


with col2:
    st.header("💬 Chat with MediGuide")

    if not st.session_state.get('kb_ready'):
        if st.session_state.get('kb_building'):
             st.info("Preparing MediGuide Assistant...")
        elif st.session_state.get('analysis_complete'):
             st.info("MediGuide will be available after Knowledge Base preparation is complete.")
        else:
             st.info("MediGuide will be available here once the report is analyzed and the knowledge base is prepared.")
    else:
        # Display chat messages from history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input using st.chat_input
        if prompt := st.chat_input("Ask about your analyzed report..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response
            if st.session_state.chat_agent_executor:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    try:
                        # Prepare simple chat history (adapt if agent needs specific format)
                        langchain_chat_history = []
                        for msg in st.session_state.chat_messages[:-1]: # Exclude current prompt
                             # Simple format, adjust if using BaseMessage objects
                             langchain_chat_history.append(f"{msg['role']}: {msg['content']}")

                        st.write("Debug: Invoking chat agent...") # Debug
                        response = st.session_state.chat_agent_executor.invoke({
                            "input": prompt,
                            "chat_history": langchain_chat_history # Pass history
                        })
                        response_content = response.get("output", "*Sorry, I encountered an issue processing that.*")
                        st.write("Debug: Chat agent invocation complete.") # Debug

                    except Exception as e:
                        st.error(f"Error during chat generation: {e}")
                        st.exception(e)
                        response_content = "*Sorry, an error occurred while generating the response.*"

                    message_placeholder.markdown(response_content) # Update placeholder with actual response
                st.session_state.chat_messages.append({"role": "assistant", "content": response_content})
            else:
                st.error("Chat agent is not available.")


# --- Footer ---
st.divider()
st.caption("MediCronus POC | Powered by CrewAI, LangChain, Groq, Neo4j, Streamlit")
