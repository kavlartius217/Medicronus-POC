# app.py
# Consolidated script for MediCronus POC

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

# Apply nest_asyncio for environments like Streamlit that might have an event loop
try:
    nest_asyncio.apply()
except RuntimeError:
    # Handle cases where it might already be applied or not needed
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Example for Windows if needed
    nest_asyncio.apply()
except Exception as e:
     st.warning(f"Could not apply nest_asyncio: {e}")
     # Fallback or proceed without if not strictly necessary for all parts

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

    @agent
    def report_extractor_agent(self) -> Agent:
        # Configure the LLM for the agent if needed (defaults to OpenAI GPT-4 if not specified)
        # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY) # Example
        return Agent(
            role="Extract, organize, and structure report contents into a clear and well-formatted document.",
            goal="Produce a logically structured and comprehensive report that enhances readability while preserving all information.",
            backstory="You are an expert in report structuring, ensuring that extracted content is well-organized, clearly formatted, and easy to navigate. Your role is to transform raw report data into a polished document with logical flow and coherence.",
            memory=True, verbose=True, allow_delegation=False#, llm=llm # Assign LLM if specific one needed
        )

    @agent
    def report_explanation_agent(self) -> Agent:
        # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY) # Example
        return Agent(
            role="Analyze the report received from the report_extractor_agent, evaluate all values, and determine which fall within the normal range and which deviate.",
            goal="Generate a well-structured and comprehensive report that classifies values as normal or abnormal, explains their significance, and identifies potential health implications based *only* on the provided report data.",
            backstory="You are an experienced pathologist specializing in interpreting blood reports with deep expertise in analyzing medical values. Your role is to evaluate, explain, and provide meaningful insights to help understand the report findings effectively. You do not provide external medical advice, only interpret the given data.",
            memory=True, verbose=True, allow_delegation=False#, llm=llm
        )

    @agent
    def abnormalities_agent(self) -> Agent:
        # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY) # Example
        return Agent(
            role="Analyze the report received from the report_explanation_agent, identify abnormal values, and focus on those. "
            "Provide potential actionable recommendations for lifestyle changes or diet based *only* on the identified abnormalities in the report. "
            "Assess the potential severity of abnormalities based on deviation from normal ranges and determine if medical consultation is generally advisable for such findings.",
            goal="Generate a structured and detailed report that highlights abnormal values, explains their significance based on the report context, "
            "and suggests general, non-prescriptive recommendations for discussion with a healthcare professional. Flag findings that typically warrant medical attention.",
            backstory="You are a highly experienced medical analyst specializing in identifying potential health risks from reports and providing data-driven insights. "
            "Your expertise lies in interpreting abnormal health metrics and offering general guidance on areas for potential improvement or further discussion with a doctor. "
            "You ensure that users understand the report's findings and the importance of professional medical consultation.",
            memory=True, verbose=True, allow_delegation=False#, llm=llm
        )

    @task
    def report_extractor_task(self) -> Task:
        return Task(
            description="Extract and organize the contents of the provided medical report: '{report}'. "
            "Ensure logical flow, clarity, and readability by using appropriate headings, subheadings, and formatting. Preserve all numerical values and test names accurately.",
            expected_output="A complete and professionally structured report in Markdown format (.md) that maintains all essential information while improving organization "
            "and presentation. The final document should be clear, concise, and easy to navigate.",
            # output_file="report.md", # CrewAI handles file naming if multiple runs occur, let's rely on the final result object
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
            # output_file="explained_report.md",
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
             # output_file="abnormalities.md" # Rely on final result object
        )

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

    @agent
    def doctor_finder_agent(self) -> Agent:
        # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY) # Example
        return Agent(
            role="Specialist Doctor Finder Agent",
            goal="Find highly-rated doctors specializing in the specific medical conditions or abnormalities provided, within the user's specified city or nearby areas. Provide detailed contact and practice information.",
            backstory="You are an AI-powered healthcare assistant equipped with advanced web search capabilities (Serper). "
            "Your purpose is to accurately identify relevant medical specialists (like Hematologists, Endocrinologists, Cardiologists, etc.) based on the patient's specific health report abnormalities and their location. "
            "You prioritize providing actionable information to help the user connect with potential doctors.",
            tools=[SerperDevTool()], # Use the search tool
            verbose=True,
            memory=False, # This agent likely doesn't need long-term memory for a single search
            allow_delegation=False#, llm=llm
        )

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
            # output_file="doctors.md" # Rely on final result object
        )

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
    # Ensure API key is available
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Cannot initialize embeddings model.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings: {e}")
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
        # Alternatives:
        # return ChatGroq(temperature=0.1, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)
        # return ChatGroq(temperature=0.1, model_name="gemma-7b-it", groq_api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")
        return None

# @st.cache_resource # Cache the connection if credentials don't change often
def setup_vector_store_and_retriever(docs_to_index: list, _embeddings):
    """Loads documents, creates Neo4j vector store, and returns a retriever tool."""
    if not docs_to_index or not _embeddings:
        st.warning("No documents or embeddings model provided for vector store setup.")
        return None, None

    # Ensure Neo4j credentials are valid
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        st.error("Neo4j connection details missing. Cannot create vector store.")
        return None, None

    st.write(f"Preparing to index {len(docs_to_index)} document chunks...") # Debug

    try:
        # Attempt to clear existing data *before* creating the new index
        # This prevents adding duplicate data if the same report is processed again
        # Note: This deletes ALL nodes with the default label ('Chunk') in the 'vector' index.
        # Be cautious if using the same Neo4j instance for other purposes.
        try:
            temp_db_to_clear = Neo4jVector.from_existing_index(
                embedding=_embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="vector", # Ensure this matches the index name used below
                database="neo4j", # Default is 'neo4j'
            )
            # You might want a more specific deletion strategy if needed
            # E.g., delete based on a session ID or filename stored as metadata
            temp_db_to_clear.delete(delete_all=True)
            st.info("Cleared existing data in Neo4j 'vector' index for this session.")
        except Exception as e:
            # This often happens if the index doesn't exist yet (first run)
            st.warning(f"Could not clear Neo4j index (may be empty or first run): {e}")

        # Create the vector store with the new documents
        db = Neo4jVector.from_documents(
            documents=docs_to_index, # Pass the already split documents
            embedding=_embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",  # Specify the database name
            index_name="vector", # Specify an index name (important for retrieval and clearing)
            # node_label="ReportChunk", # Optional: Custom node label
            # embedding_node_property="embedding", # Default property name
            # text_node_property="text", # Default property name
        )
        st.success("Neo4j Vector Store created/updated successfully!")
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            "Patient_Report_Analysis_Tool", # Tool name
            "Use this tool to search for information within the patient's analyzed medical report. It contains the structured report, explanation of terms and values, identified abnormalities, and potentially doctor recommendations." # Tool description
        )
        return tool, db
    except Exception as e:
        st.error(f"Failed to create/update Neo4j Vector Store: {e}")
        # Provide more detail if possible, e.g., check connection, credentials, embedding dimensions
        return None, None

# @st.cache_resource # Cache the agent executor
def create_chat_agent(_llm, _tool):
    """Creates the Langchain AgentExecutor for the chatbot."""
    if _tool is None or _llm is None:
        st.error("LLM or Retriever Tool is missing. Cannot create chat agent.")
        return None

    # Define the prompt for the chat agent
    # Use MessagesPlaceholder for history and agent_scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are MediGuide, a helpful AI assistant designed to discuss the patient's *most recently analyzed* medical report. Your knowledge comes solely from the 'Patient_Report_Analysis_Tool'. Answer questions based *only* on the information retrieved by this tool. Explain the report's contents, abnormalities, and any recommendations found within the report. Do *not* provide external medical advice, diagnoses, or opinions. If asked about topics outside the provided report context, state that you can only discuss the analyzed document."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") # Necessary for tool use / agent thoughts
    ])

    try:
        # Create the agent using the LLM, tool, and prompt
        agent = create_openai_tools_agent(llm=_llm, tools=[_tool], prompt=prompt)
        # Create the AgentExecutor
        agent_exec = AgentExecutor(
            agent=agent,
            tools=[_tool],
            verbose=True, # Set to True for debugging agent steps
            handle_parsing_errors=True, # Attempt to gracefully handle LLM output parsing errors
            max_iterations=5 # Prevent runaway agents
            )
        st.success("Chat agent created successfully.")
        return agent_exec
    except Exception as e:
        st.error(f"Failed to create chat agent: {e}")
        return None

# --- File Handling & Text Processing ---
def read_file_content(file_path):
    """Reads content from a file path."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return None # Return None if file doesn't exist yet
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return None

def split_text_for_indexing(text_content: str):
    """Splits text content into chunks suitable for embedding."""
    if not text_content:
        return []
    rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75) # Adjusted overlap
    # Use create_documents to add basic metadata if needed, or just split_text
    # docs = rcts.create_documents([text_content])
    docs = rcts.split_text(text_content)
    # Convert simple text chunks into Langchain Document objects for Neo4jVector
    from langchain_core.documents import Document
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
        # Display uploaded file name
        st.write(f"Uploaded: **{uploaded_file.name}**")
        st.session_state.current_pdf_name = uploaded_file.name

        # Check if this is a new file or analysis needs to be run
        if st.session_state.uploaded_file_path is None or Path(st.session_state.uploaded_file_path).name != uploaded_file.name:
            # Save the new file temporarily
            temp_dir = Path("./temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            # Use a unique name derived from the original? Or just UUID? Using UUID is safer.
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            temp_pdf_path = temp_dir / unique_filename

            try:
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = str(temp_pdf_path)
                st.success("File ready for analysis.")
                # Reset status for the new file
                st.session_state.analysis_complete = False
                st.session_state.doctors_complete = False
                st.session_state.kb_ready = False
                st.session_state.chat_agent_executor = None
                st.session_state.chat_messages = []
                st.session_state.analysis_results = {}
                st.session_state.doctor_recommendations = None
                st.session_state.user_wants_doctors = None
                st.session_state.user_city = ""

            except Exception as e:
                st.error(f"Error saving uploaded file: {e}")
                st.session_state.uploaded_file_path = None

    # Button to trigger analysis - only active if a file is uploaded and not already analyzed
    if st.session_state.uploaded_file_path and not st.session_state.analysis_complete:
        if st.button("2. Analyze Report", key="analyze_button", disabled=st.session_state.analysis_running):
            st.session_state.analysis_running = True
            st.session_state.analysis_complete = False # Ensure reset before starting
            st.session_state.doctors_complete = False
            st.session_state.kb_ready = False
            st.session_state.chat_agent_executor = None
            st.session_state.chat_messages = []

            with st.spinner("Analyzing report... This may take a few moments."):
                try:
                    st.write("Loading PDF content...")
                    loader = PyPDFLoader(st.session_state.uploaded_file_path)
                    report_docs = loader.load()
                    report_content = "\n\n".join([doc.page_content for doc in report_docs])

                    if not report_content.strip():
                         st.error("Failed to extract text content from the PDF.")
                         raise ValueError("Empty report content")

                    st.write("Running Medic Bot Crew...")
                    medic_crew = Medic_Bot().crew()
                    # Ensure inputs dict keys match task descriptions' placeholders
                    result = medic_crew.kickoff(inputs={"report": report_content})

                    # Process the result - expecting markdown outputs from tasks
                    # CrewAI's result object might structure outputs differently.
                    # Assuming the final result aggregates task outputs or the last task's output is primary.
                    # Let's try to access specific task outputs if possible (depends on CrewAI version/structure)
                    st.session_state.analysis_results = {
                        "structured_report": result.tasks_output[0].raw_output if len(result.tasks_output) > 0 else "Not generated.",
                        "explained_report": result.tasks_output[1].raw_output if len(result.tasks_output) > 1 else "Not generated.",
                        "abnormalities_report": result.tasks_output[2].raw_output if len(result.tasks_output) > 2 else "Not generated."
                        # Fallback to overall raw if specific task outputs aren't easily accessible
                        # "abnormalities_report": result.raw # Assuming the last task output is the raw result
                    }

                    if not st.session_state.analysis_results.get("abnormalities_report"):
                         st.warning("Could not extract abnormalities report from crew result.")
                         # Maybe use the final raw output as a fallback
                         st.session_state.analysis_results["abnormalities_report"] = result.raw


                    st.session_state.analysis_complete = True
                    st.success("Report analysis complete!")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.session_state.analysis_complete = False
                finally:
                    st.session_state.analysis_running = False
                    # Clean up temp file? Optional, might keep for debugging or re-analysis
                    # if st.session_state.uploaded_file_path and Path(st.session_state.uploaded_file_path).exists():
                    #     Path(st.session_state.uploaded_file_path).unlink()
                    #     st.session_state.uploaded_file_path = None # Reset path after processing
                    st.rerun() # Rerun to update UI based on completion status

    # Display current status
    st.sidebar.divider()
    st.sidebar.write("Status:")
    if st.session_state.analysis_complete:
        st.sidebar.success("✅ Analysis Complete")
    else:
        st.sidebar.info("⏳ Analysis Pending")

    if st.session_state.user_wants_doctors == 'Yes' and st.session_state.doctors_complete:
         st.sidebar.success("✅ Doctors Searched")
    elif st.session_state.user_wants_doctors == 'No':
         st.sidebar.info("⚪ Doctors Skipped")
    elif st.session_state.user_wants_doctors == 'Yes':
         st.sidebar.info("⏳ Doctor Search Pending")

    if st.session_state.kb_ready:
        st.sidebar.success("✅ MediGuide Ready")
    else:
        st.sidebar.info("⏳ MediGuide Not Ready")


# --- Main Area Layout (Report Display & Chat) ---
col1, col2 = st.columns([3, 2]) # Give more space to report display

with col1:
    st.header("Analysis Results")
    if not st.session_state.analysis_complete:
        if st.session_state.uploaded_file_path:
             st.info("Report uploaded. Click 'Analyze Report' in the sidebar to begin.")
        else:
             st.info("Upload a PDF report using the sidebar to start the process.")
    else:
        # --- Display Analysis Results in Tabs ---
        tab1, tab2, tab3 = st.tabs(["📄 Structured Report", "🩺 Explained Report", "❗ Abnormalities Summary"])

        with tab1:
            report_md = st.session_state.analysis_results.get("structured_report", "No structured report generated.")
            st.markdown(report_md)
        with tab2:
            explained_md = st.session_state.analysis_results.get("explained_report", "No explained report generated.")
            st.markdown(explained_md)
        with tab3:
            abnormal_md = st.session_state.analysis_results.get("abnormalities_report", "No abnormalities summary generated.")
            st.markdown(abnormal_md)

        st.divider()

        # --- Doctor Recommendation Flow ---
        if not st.session_state.analysis_results.get("abnormalities_report"):
            st.warning("Cannot search for doctors as abnormalities report was not generated.")
        elif st.session_state.user_wants_doctors is None: # Ask only once per analysis
             st.subheader("Find Specialists?")
             st.radio(
                 "Would you like to search for recommended doctors based on the abnormalities?",
                 ("Yes", "No"),
                 key="want_doctors_radio",
                 index=None,
                 on_change=lambda: setattr(st.session_state, 'user_wants_doctors', st.session_state.want_doctors_radio)
             )

        # If user wants doctors and search hasn't been done/completed yet
        if st.session_state.user_wants_doctors == "Yes" and not st.session_state.doctors_complete:
            st.subheader("Enter Location")
            city = st.text_input("Your City:", key="city_input", value=st.session_state.user_city)
            st.session_state.user_city = city # Store city input

            if city:
                if st.button("3. Find Doctors", key="find_doctors_button", disabled=st.session_state.doctors_running):
                    st.session_state.doctors_running = True
                    st.session_state.doctors_complete = False # Reset before starting search

                    with st.spinner("Searching for relevant doctors..."):
                        try:
                            st.write(f"Searching doctors in {city}...")
                            doctor_crew = Doctor_Bot().crew()
                            # Pass necessary inputs to the doctor crew
                            abnormalities_summary = st.session_state.analysis_results.get("abnormalities_report", "General Health Checkup")
                            # Maybe extract just the list of abnormalities if the report is structured that way
                            inputs = {
                                "user_city": city,
                                "abnormalities": abnormalities_summary # Pass the abnormalities text/summary
                            }
                            doctor_result = doctor_crew.kickoff(inputs=inputs)
                            # Assuming the result is the markdown text in the raw output
                            st.session_state.doctor_recommendations = doctor_result.raw
                            st.session_state.doctors_complete = True # Mark as complete (search attempted)
                            st.success("Doctor search finished.")

                        except Exception as e:
                            st.error(f"An error occurred during doctor search: {e}")
                            st.session_state.doctors_complete = False # Mark as failed
                        finally:
                            st.session_state.doctors_running = False
                            st.rerun() # Update UI

        # Display doctor results if search was completed
        if st.session_state.doctors_complete and st.session_state.doctor_recommendations:
            st.subheader("Doctor Recommendations")
            with st.expander("View Found Specialists", expanded=True):
                 st.markdown(st.session_state.doctor_recommendations)
        elif st.session_state.doctors_complete and not st.session_state.doctor_recommendations:
             st.info("The search completed, but no specific doctor recommendations were found or generated.")

        # Acknowledge if user chose No
        if st.session_state.user_wants_doctors == "No":
            st.info("Skipping doctor recommendations.")


        # --- Knowledge Base Building (Trigger Automatically after Analysis/Doctor steps) ---
        # Determine if ready for KB: analysis done AND (doctors skipped OR doctors searched)
        ready_for_kb = st.session_state.analysis_complete and \
                       (st.session_state.user_wants_doctors == 'No' or st.session_state.doctors_complete)

        if ready_for_kb and not st.session_state.kb_ready and not st.session_state.kb_building:
            st.session_state.kb_building = True
            with st.spinner("Preparing MediGuide Assistant... Building Knowledge Base"):
                try:
                    st.write("Gathering analyzed content...")
                    # Combine relevant text content for indexing
                    content_to_index = []
                    if st.session_state.analysis_results.get("structured_report"):
                        content_to_index.append(st.session_state.analysis_results["structured_report"])
                    if st.session_state.analysis_results.get("explained_report"):
                        content_to_index.append(st.session_state.analysis_results["explained_report"])
                    if st.session_state.analysis_results.get("abnormalities_report"):
                        content_to_index.append(st.session_state.analysis_results["abnormalities_report"])
                    if st.session_state.doctor_recommendations:
                        content_to_index.append(st.session_state.doctor_recommendations)

                    full_text = "\n\n---\n\n".join(content_to_index)

                    if not full_text.strip():
                        st.error("No content available to build knowledge base.")
                        raise ValueError("Empty content for KB")

                    st.write("Splitting documents...")
                    doc_chunks = split_text_for_indexing(full_text)

                    if not doc_chunks:
                         st.error("Failed to split documents for indexing.")
                         raise ValueError("No document chunks created")

                    st.write("Setting up vector store...")
                    embeddings = get_embeddings_model()
                    if embeddings:
                         retriever_tool, vector_db = setup_vector_store_and_retriever(doc_chunks, embeddings)

                         if retriever_tool:
                            st.write("Initializing chat agent...")
                            llm = get_llm()
                            if llm:
                                agent_executor = create_chat_agent(llm, retriever_tool)
                                if agent_executor:
                                    st.session_state.chat_agent_executor = agent_executor
                                    st.session_state.kb_ready = True
                                    st.session_state.chat_messages = [] # Clear history for new KB
                                    st.success("MediGuide Assistant is ready!")
                                else:
                                     st.error("Failed to create chat agent executor.")
                            else:
                                st.error("Failed to initialize LLM for chat agent.")
                         else:
                             st.error("Failed to setup retriever tool for knowledge base.")
                    else:
                        st.error("Failed to initialize embeddings model.")

                except Exception as e:
                    st.error(f"Failed to build knowledge base or setup chatbot: {e}")
                    st.session_state.kb_ready = False
                finally:
                     st.session_state.kb_building = False
                     st.rerun() # Update UI


with col2:
    st.header("💬 Chat with MediGuide")

    if not st.session_state.kb_ready:
        st.info("MediGuide will be available here once the report is analyzed and the knowledge base is prepared.")
        if st.session_state.kb_building:
            st.warning("Currently preparing MediGuide...")
    else:
        # Display chat messages from history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input using st.chat_input
        if prompt := st.chat_input("Ask about your analyzed report..."):
            # Add user message to state and display it
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response
            if st.session_state.chat_agent_executor:
                with st.chat_message("assistant"):
                    with st.spinner("MediGuide is thinking..."):
                        try:
                            # Prepare the chat history in the format expected by the agent
                            # (Often list of BaseMessage objects or simple key-value pairs)
                            # This example assumes the agent handles a list of dicts via MessagesPlaceholder conversion
                            langchain_chat_history = []
                            for msg in st.session_state.chat_messages[:-1]: # Exclude the current prompt
                                langchain_chat_history.append(f"{msg['role']}: {msg['content']}")

                            response = st.session_state.chat_agent_executor.invoke({
                                "input": prompt,
                                "chat_history": langchain_chat_history
                            })
                            response_content = response.get("output", "Sorry, I encountered an issue processing that.")

                        except Exception as e:
                            st.error(f"Error during chat generation: {e}")
                            response_content = "Sorry, an error occurred while generating the response."

                    st.markdown(response_content)
                # Add assistant response to state
                st.session_state.chat_messages.append({"role": "assistant", "content": response_content})
            else:
                st.error("Chat agent is not available.")


# --- Footer ---
st.divider()
st.caption("MediCronus POC | Powered by CrewAI, LangChain, Groq, Neo4j, Streamlit")
