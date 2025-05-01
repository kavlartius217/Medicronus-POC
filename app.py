__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import nest_asyncio
import traceback
import time
from crewai import Agent, Task, Crew
from crewai.project import agent, task, crew, CrewBase
# Removed unused flow imports: Flow, start, listen, and_, or_, router
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from pydantic import BaseModel # Removed State as it wasn't used in the final flow approach

# Apply nest_asyncio for compatibility with Streamlit
nest_asyncio.apply()

# --- Configuration & Setup ---

# Set page configuration
st.set_page_config(
    page_title="Medicronus MARK I",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger (simple print-based for this example)
def log_error(error):
    """Log error to Streamlit and print full traceback to console"""
    st.error(f"An error occurred: {str(error)}")
    error_trace = traceback.format_exc()
    print(f"--- ERROR ---")
    print(f"Error Message: {str(error)}")
    print(f"Traceback:\n{error_trace}")
    print(f"--- END ERROR ---")
    # Optional: Add file logging here if needed

# Sidebar for API keys
with st.sidebar:
    st.image("https://i.ibb.co/M8JQs2v/medicronus-logo.png", width=250)
    st.title("Medicronus MARK I")
    st.subheader("Healthcare AI Assistant")

    with st.expander("⚙️ API Configuration", expanded=False):
        # Use session state to preserve API key inputs across reruns if needed
        serper_api = st.text_input("SERPER_DEV_TOOL API Key", type="password", key="serper_api_key_input")
        openai_api = st.text_input("OpenAI API Key", type="password", key="openai_api_key_input")
        groq_api = st.text_input("Groq API Key", type="password", key="groq_api_key_input")
        neo4j_uri = st.text_input("Neo4j URI", value="neo4j+s://your_neo4j_instance.databases.neo4j.io", key="neo4j_uri_input") # Replace default
        neo4j_username = st.text_input("Neo4j Username", value="neo4j", key="neo4j_username_input")
        neo4j_password = st.text_input("Neo4j Password", type="password", key="neo4j_password_input")

        if st.button("Save API Keys"):
            missing_keys = []
            if not serper_api: missing_keys.append("SERPER_DEV_TOOL")
            if not openai_api: missing_keys.append("OpenAI")
            if not groq_api: missing_keys.append("Groq")
            if not neo4j_uri: missing_keys.append("Neo4j URI")
            if not neo4j_username: missing_keys.append("Neo4j Username")
            if not neo4j_password: missing_keys.append("Neo4j Password")

            if missing_keys:
                st.warning(f"Please fill in all required fields: {', '.join(missing_keys)}")
            else:
                try:
                    os.environ['SERPER_API_KEY'] = serper_api # Correct env var name for Serper tool often
                    os.environ['SERPER_DEV_TOOL'] = serper_api # Keep this too just in case
                    os.environ['OPENAI_API_KEY'] = openai_api
                    os.environ['GROQ_API_KEY'] = groq_api
                    os.environ['NEO4J_URI'] = neo4j_uri
                    os.environ['NEO4J_USERNAME'] = neo4j_username
                    os.environ['NEO4J_PASSWORD'] = neo4j_password
                    st.success("API keys saved to environment variables for this session!")
                    print("[INFO] API keys saved to environment variables.")
                except Exception as e:
                    log_error(e)
                    st.error("Failed to save API keys.")

# Create temporary directory and placeholder files safely
if 'temp_dir' not in st.session_state:
    try:
        st.session_state.temp_dir = tempfile.mkdtemp()
        print(f"[INFO] Created temporary directory: {st.session_state.temp_dir}")
        # Create placeholder files to avoid FileNotFoundError during first run or failures
        placeholder_content = "Placeholder content. This file should be populated during analysis."
        for file_name in ["report.md", "explained_report.md", "abnormalities.md", "doctors.md"]:
            try:
                with open(os.path.join(st.session_state.temp_dir, file_name), "w", encoding='utf-8') as f:
                    f.write(placeholder_content)
            except Exception as e_file:
                print(f"[ERROR] Failed to create placeholder file {file_name}: {e_file}")
                # Let the app continue, but warn the user
                st.warning(f"Could not create placeholder file {file_name}. App might encounter issues.")

    except Exception as e:
        log_error(e)
        st.error("CRITICAL: Failed to create temporary directory. App cannot function.")
        # Optionally stop the app if temp dir is essential: st.stop()
        st.session_state.temp_dir = None # Indicate failure

# Initialize session state variables
default_states = {
    'report_content': None,
    'extracted_report': "Analysis not yet run or file not generated.",
    'explained_report': "Analysis not yet run or file not generated.",
    'abnormalities': "Analysis not yet run or file not generated.",
    'doctors_list': "Doctor search not yet run or file not generated.",
    'chat_history': [],
    'analysis_complete': False,
    'doctor_search_complete': False,
    'agent_executor': None # Store the initialized chat agent
}
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- CrewAI Definitions ---

# Define the CrewBase classes
@CrewBase
class Medic_Bot():
    """The Medic Bot analyzes the health report and generates comprehensive insights."""

    @agent
    def report_extractor_agent(self) -> Agent:
        return Agent(
            role="Extract, organize, and structure report contents into a clear and well-formatted markdown document.",
            goal="Produce a logically structured and comprehensive report in markdown format that enhances readability while preserving all information.",
            backstory="You are an expert in report structuring, ensuring that extracted content is well-organized, clearly formatted using markdown, and easy to navigate. Your role is to transform raw report data into a polished document with logical flow and coherence.",
            memory=True,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def report_explanation_agent(self) -> Agent:
        return Agent(
            role="Analyze the structured report, evaluate all values, determine normal/abnormal ranges, and explain significance.",
            goal="Generate a well-structured markdown report that classifies values as normal or abnormal, explains their significance, and assigns severity.",
            backstory="You are an experienced pathologist specializing in interpreting blood reports with deep expertise in analyzing medical values. Your role is to evaluate, explain using markdown formatting, and provide meaningful insights to help understand the report findings effectively.",
            memory=True,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def abnormalities_agent(self) -> Agent:
        return Agent(
            role="Identify abnormalities, explain risks, provide actionable recommendations, assess severity, and suggest specialist types.",
            goal="Generate a structured markdown report detailing abnormal values, their significance, practical recommendations (lifestyle, diet, medical), severity assessment, and specialist suggestions.",
            backstory="You are a highly experienced medical analyst specializing in identifying health risks from reports and providing data-driven recommendations using markdown. Your expertise lies in interpreting abnormal health metrics and offering practical guidance. You ensure users understand their status and needed actions.",
            memory=True,
            verbose=True,
            allow_delegation=False
        )

    @task
    def report_extractor_task(self) -> Task:
        # Ensure the temp_dir is available when the task is defined
        if not st.session_state.get('temp_dir'):
             raise ValueError("Temporary directory not available for task definition.")
        return Task(
            description=(
                "Extract and organize the contents of the provided medical report: '{report}'. "
                "Format the output as a well-structured, coherent, and comprehensive markdown document. "
                "Use appropriate markdown headings (#, ##), subheadings, lists (* or -), and bold text (**bold**) for clarity and readability."
            ),
            expected_output=(
                "A complete and professionally structured markdown report that maintains all essential information while improving organization "
                "and presentation. The final document should be clear, concise, and easy to navigate using markdown formatting."
            ),
            output_file=os.path.join(st.session_state.temp_dir, "report.md"),
            agent=self.report_extractor_agent()
        )

    @task
    def report_explanation_task(self) -> Task:
        if not st.session_state.get('temp_dir'):
             raise ValueError("Temporary directory not available for task definition.")
        return Task(
            description=(
                "Analyze the structured markdown report provided by the report_extractor_agent. "
                "Evaluate all numerical and categorical values to classify them as normal or abnormal based on standard medical ranges. "
                "Assign a severity score (e.g., Low, Moderate, High) to each abnormality. "
                "Provide a detailed explanation in markdown format, including potential causes, health implications, and general medical context. "
                "Structure the report for clarity using markdown headings, subheadings, and bullet points."
            ),
            expected_output=(
                "A well-organized, detailed medical report in markdown format that:\n"
                "- Clearly classifies each significant value as Normal or Abnormal.\n"
                "- Assigns a severity score to every abnormality.\n"
                "- Provides meaningful explanations for abnormal values.\n"
                "- Is structured with clear markdown headings, subheadings, and lists for readability.\n"
            ),
            agent=self.report_explanation_agent(),
            output_file=os.path.join(st.session_state.temp_dir, "explained_report.md"),
            context=[self.report_extractor_task()] # Depends on the extractor task
        )

    @task
    def abnormalities_task(self) -> Task:
        if not st.session_state.get('temp_dir'):
             raise ValueError("Temporary directory not available for task definition.")
        return Task(
            description=(
                "Analyze the explained report from the report_explanation_agent, focusing exclusively on the identified abnormal values. "
                "For each abnormal metric: determine potential health risks, explain why it is outside the normal range, "
                "and provide specific, actionable recommendations (lifestyle changes, dietary modifications, potential treatments - use general terms). "
                "Assess the severity of each abnormality again if needed and indicate whether medical consultation is advised. "
                "Suggest the type of medical specialist (e.g., Endocrinologist, Hematologist) relevant to the abnormalities found. "
                "Format the output clearly using markdown."
            ),
            expected_output=(
                "A structured markdown abnormalities report that:\n"
                "- Lists all abnormal values with their medical significance.\n"
                "- Explains potential causes and health implications.\n"
                "- Provides actionable, personalized recommendations (using markdown lists/bolding).\n"
                "- Confirms severity score (Low / Moderate / High) with justification.\n"
                "- Recommends the type(s) of medical specialist(s) to consult.\n"
                "- Flags any findings suggesting urgent medical attention.\n"
                "- Is formatted with clear markdown headings and lists for usability."
            ),
            context=[self.report_explanation_task()], # Depends on the explanation task
            agent=self.abnormalities_agent(),
            output_file=os.path.join(st.session_state.temp_dir, "abnormalities.md")
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.report_extractor_agent(), self.report_explanation_agent(), self.abnormalities_agent()],
            tasks=[self.report_extractor_task(), self.report_explanation_task(), self.abnormalities_task()],
            verbose=2 # Crew verbose level
        )

@CrewBase
class Doctor_Bot():
    """This bot finds relevant doctors based on abnormalities."""

    @agent
    def doctor_finder_agent(self) -> Agent:
        # Ensure API key is available for the tool
        if not os.environ.get('SERPER_API_KEY'):
             print("[WARN] SERPER_API_KEY environment variable not found for Doctor Finder Agent tool.")
             # Agent might still be created but tool will fail later
        return Agent(
            role="Doctor Finder Agent",
            goal="Find suitable doctors specializing in the detected abnormalities near the user's location and provide their details.",
            backstory=(
                "You are an AI healthcare assistant equipped with web search capabilities. Based on a list of medical abnormalities "
                "and a user-provided city, you search for relevant specialists (like hematologists, endocrinologists, cardiologists) "
                "and compile a list of potential doctors, formatted in markdown."
            ),
            tools=[SerperDevTool(api_key=os.environ.get('SERPER_API_KEY'))] if os.environ.get('SERPER_API_KEY') else [], # Only add tool if key exists
            verbose=True,
            memory=True,
            allow_delegation=False
        )

    @task
    def doctor_finder_task(self) -> Task:
        if not st.session_state.get('temp_dir'):
             raise ValueError("Temporary directory not available for task definition.")
        return Task(
            description=(
                "Search for top-rated doctors specializing in the medical conditions related to these abnormalities: '{abnormalities}'. "
                "Focus the search on doctors located in or very near the city: '{user_city}'.\n"
                "Compile a list formatted in markdown. For each recommended specialist, provide:\n"
                "1.  **Doctor's Name**\n"
                "2.  **Specialization** (e.g., Endocrinologist, Hematologist)\n"
                "3.  **Clinic/Hospital Name** (if available)\n"
                "4.  **Location/Address** (at least city/area)\n"
                "5.  **Contact Information** (Phone/Website if found)\n"
                "6.  **Brief Note** (e.g., focus area, rating if easily found - optional)"
            ),
            expected_output=(
                "A markdown-formatted list of specialists matching the criteria. Each entry should clearly present the "
                "doctor's name, specialization, location, and contact details if found. Handle cases where few or no doctors are found gracefully."
            ),
            agent=self.doctor_finder_agent(),
            output_file=os.path.join(st.session_state.temp_dir, "doctors.md")
            # Note: Context (abnormalities) is passed via kickoff dictionary, not task context here
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.doctor_finder_agent()],
            tasks=[self.doctor_finder_task()],
            verbose=2
        )

# --- Core Logic Class ---

class MediTrustAI():
    """Handles the workflow orchestration for analysis and doctor search."""

    def run_analysis(self, report_content):
        """Runs the Medic_Bot crew and handles file reading."""
        if not st.session_state.get('temp_dir'):
            st.error("Temporary directory is not available. Cannot run analysis.")
            return False # Indicate failure

        st.session_state.analysis_complete = False # Reset flag
        try:
            print("[INFO] Initializing Medic_Bot...")
            medic_crew = Medic_Bot().crew()
            print("[INFO] Starting Medic_Bot crew kickoff...")
            # Pass report content as input to the first task implicitly via crew input dict
            result = medic_crew.kickoff(inputs={'report': report_content})
            print(f"[INFO] Medic_Bot kickoff finished. Raw result: {result}") # CrewAI result object

            # --- Enhanced File Reading ---
            files_to_read = {
                "extracted_report": "report.md",
                "explained_report": "explained_report.md",
                "abnormalities": "abnormalities.md"
            }
            temp_dir = st.session_state.temp_dir
            print(f"[DEBUG] Reading analysis files from: {temp_dir}")

            all_reads_successful = True

            for state_key, filename in files_to_read.items():
                file_path = os.path.join(temp_dir, filename)
                print(f"[DEBUG] --- Attempting to read: {file_path} for state_key: '{state_key}' ---")
                if os.path.exists(file_path):
                    print(f"[DEBUG] File exists: {file_path}")
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            content = f.read()

                        if not content.strip() or "Placeholder content" in content:
                             print(f"[WARN] File '{filename}' seems empty or contains placeholder.")
                             # Decide if this counts as failure
                             # For critical files like abnormalities, maybe count as failure:
                             if state_key == 'abnormalities':
                                 print(f"[ERROR] Critical file '{filename}' is empty/placeholder. Marking read as failed.")
                                 st.session_state[state_key] = f"{state_key.replace('_', ' ').title()} file is empty or was not properly generated."
                                 all_reads_successful = False
                             else:
                                 st.session_state[state_key] = content # Store empty/placeholder content
                                 print(f"[DEBUG] Stored empty/placeholder content for '{state_key}'.")
                        else:
                            # Assign ONLY IF read was successful and content is valid
                            st.session_state[state_key] = content
                            print(f"[DEBUG] Successfully read {len(content)} chars into st.session_state['{state_key}']. Preview: '{content[:100].strip()}...'")

                    except Exception as e:
                        print(f"[ERROR] FAILED to read file {file_path}: {e}")
                        log_error(f"Reading {filename}: {e}") # Log the error
                        st.session_state[state_key] = f"Error reading content from {filename}."
                        all_reads_successful = False
                else:
                    print(f"[ERROR] File NOT FOUND: {file_path}")
                    st.warning(f"Analysis output file not found: {filename}.")
                    st.session_state[state_key] = f"{state_key.replace('_', ' ').title()} file was not generated."
                    all_reads_successful = False
                print(f"[DEBUG] --- Finished reading attempt for: {filename} ---")

            # Set completion flag based ONLY on successful reads (adjust criteria as needed)
            # Crucially, ensure 'abnormalities' was read successfully if it's needed for next steps
            if all_reads_successful and st.session_state.get("abnormalities", "").strip() and "Placeholder content" not in st.session_state.get("abnormalities", ""):
                print("[INFO] All expected analysis files read successfully. Setting analysis_complete = True")
                st.session_state.analysis_complete = True
                return True
            else:
                print("[ERROR] One or more analysis files failed critical read checks. Setting analysis_complete = False")
                st.session_state.analysis_complete = False
                st.error("Analysis completed, but failed to retrieve or validate all necessary report sections.")
                return False

        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during Medic_Bot execution or file processing:")
            log_error(e) # Log the full error
            st.session_state.analysis_complete = False
            return False

    def run_doctor_search(self, user_city, abnormalities_content):
        """Runs the Doctor_Bot crew and handles file reading."""
        if not st.session_state.get('temp_dir'):
            st.error("Temporary directory is not available. Cannot run doctor search.")
            return False

        if not os.environ.get('SERPER_API_KEY'):
            st.error("SERPER API Key not found in environment. Doctor search requires this key (set in sidebar).")
            return False

        st.session_state.doctor_search_complete = False # Reset flag
        try:
            print("[INFO] Initializing Doctor_Bot...")
            doctor_crew = Doctor_Bot().crew()
            print("[INFO] Starting Doctor_Bot crew kickoff...")
            result = doctor_crew.kickoff(inputs={
                'user_city': user_city,
                'abnormalities': abnormalities_content # Pass the abnormalities text
            })
            print(f"[INFO] Doctor_Bot kickoff finished. Raw result: {result}")

            # --- Enhanced File Reading for doctors.md ---
            doctors_filename = "doctors.md"
            doctors_key = "doctors_list"
            temp_dir = st.session_state.temp_dir
            doctors_path = os.path.join(temp_dir, doctors_filename)
            read_successful = False

            print(f"[DEBUG] --- Attempting to read: {doctors_path} for state_key: '{doctors_key}' ---")
            if os.path.exists(doctors_path):
                 print(f"[DEBUG] File exists: {doctors_path}")
                 try:
                     with open(doctors_path, "r", encoding='utf-8') as f:
                         content = f.read()

                     if not content.strip() or "Placeholder content" in content:
                         print(f"[WARN] File '{doctors_filename}' seems empty or contains placeholder.")
                         st.session_state[doctors_key] = "Doctor search ran but did not find specific recommendations or failed to write output."
                         # Consider if empty doctors list is a "success" or "failure"
                         read_successful = True # Let's say running is success, even if list empty
                     else:
                         st.session_state[doctors_key] = content
                         print(f"[DEBUG] Successfully read {len(content)} chars into st.session_state['{doctors_key}']. Preview: '{content[:100].strip()}...'")
                         read_successful = True
                 except Exception as e:
                     print(f"[ERROR] FAILED to read file {doctors_path}: {e}")
                     log_error(f"Reading {doctors_filename}: {e}")
                     st.session_state[doctors_key] = f"Error reading content from {doctors_filename}."
                     # read_successful remains False
            else:
                 print(f"[ERROR] File NOT FOUND: {doctors_path}")
                 st.warning(f"Doctors output file not found: {doctors_filename}.")
                 st.session_state[doctors_key] = "Doctors file was not generated by the search process."
                 # read_successful remains False
            print(f"[DEBUG] --- Finished reading attempt for: {doctors_filename} ---")

            # Set completion flag based on successful read
            if read_successful:
                print("[INFO] Doctor search file processed. Setting doctor_search_complete = True")
                st.session_state.doctor_search_complete = True
                return True
            else:
                print("[ERROR] Doctor search file failed read check. Setting doctor_search_complete = False")
                st.session_state.doctor_search_complete = False
                st.error("Doctor search completed, but failed to retrieve the results file.")
                return False

        except Exception as e:
            print(f"[CRITICAL ERROR] Exception during Doctor_Bot execution or file processing:")
            log_error(e)
            st.session_state.doctor_search_complete = False
            return False

# --- Chat Interface Setup ---

def setup_chat_interface():
    """Sets up the Langchain AgentExecutor for the chat tab."""
    # Only setup if analysis is complete and we have abnormalities data
    if not st.session_state.analysis_complete or not st.session_state.abnormalities or "Placeholder content" in st.session_state.abnormalities:
        print("[INFO] Chat setup skipped: Analysis not complete or abnormalities missing.")
        return None

    # Check for required API keys
    if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
        st.warning("OpenAI API Key not set. Knowledge base setup requires embeddings.")
        return None
    if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY']:
        st.warning("Groq API Key not set. Chat functionality requires this.")
        return None
    if 'NEO4J_URI' not in os.environ or 'NEO4J_USERNAME' not in os.environ or 'NEO4J_PASSWORD' not in os.environ:
        st.warning("Neo4j credentials not set. Knowledge base setup requires Neo4j.")
        return None
    if not st.session_state.get('temp_dir'):
        st.error("Temporary directory not available. Cannot load documents for chat.")
        return None


    print("[INFO] Setting up chat interface...")
    try:
        temp_dir = st.session_state.temp_dir
        # Ensure necessary files exist and contain actual content before loading
        docs_to_load = []
        file_paths = {
            "abnormalities": os.path.join(temp_dir, "abnormalities.md"),
            "explained_report": os.path.join(temp_dir, "explained_report.md"),
            "doctors": os.path.join(temp_dir, "doctors.md")
        }
        session_state_keys = {
            "abnormalities": "abnormalities",
            "explained_report": "explained_report",
            "doctors": "doctors_list"
        }

        for name, path in file_paths.items():
            content_key = session_state_keys[name]
            # Check if file exists AND corresponding session state has valid content
            if os.path.exists(path) and st.session_state.get(content_key) and "Placeholder content" not in st.session_state.get(content_key):
                 print(f"[DEBUG] Found valid content for {name}, adding {path} to loader.")
                 # Ensure file has content before loading
                 if os.path.getsize(path) > 0:
                    docs_to_load.append(TextLoader(path, encoding='utf-8'))
                 else:
                    print(f"[WARN] File {path} exists but is empty. Skipping.")
            else:
                 print(f"[DEBUG] Skipping {path} for chat context (doesn't exist or content invalid).")

        if not docs_to_load:
            st.warning("No valid analysis documents found to build knowledge base for chat.")
            return None

        # Load and split documents
        loaded_docs = []
        for loader in docs_to_load:
            try:
                loaded_docs.extend(loader.load())
            except Exception as e_load:
                print(f"[ERROR] Failed to load document {loader.file_path}: {e_load}")
                st.warning(f"Could not load {os.path.basename(loader.file_path)} for chat context.")

        if not loaded_docs:
             st.warning("Failed to load any documents into the knowledge base.")
             return None

        rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = rcts.split_documents(loaded_docs)
        print(f"[INFO] Split {len(loaded_docs)} documents into {len(split_docs)} chunks.")

        # Create Neo4j Vector Store
        print("[INFO] Initializing Neo4j vector store...")
        try:
            embeddings = OpenAIEmbeddings()
            # Consider adding index_name or node_label for better organization if needed
            # Using remove_existing=True might be useful during development to clear old data
            db = Neo4jVector.from_documents(
                embedding=embeddings,
                documents=split_docs,
                url=os.environ['NEO4J_URI'],
                username=os.environ['NEO4J_USERNAME'],
                password=os.environ['NEO4J_PASSWORD'],
                # database="neo4j", # Default db name
                # index_name="medicronus_report_index", # Optional: specify index name
                # node_label="ReportChunk", # Optional: specify node label
                # remove_existing=True # Use with caution - deletes data with same label/index
            )
            print("[INFO] Neo4j vector store created/updated.")
        except Exception as e_neo:
            print(f"[CRITICAL ERROR] Failed to initialize or populate Neo4j Vector Store:")
            log_error(e_neo)
            st.error("Failed to connect to or update the knowledge base (Neo4j). Chat may not work correctly.")
            return None

        # Create Retriever Tool
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            "Patient_Records_Context_Tool", # More specific name
            ("Provides context about the patient's current medical report analysis. Use this tool to answer questions "
             "specifically about the findings, explanations, abnormalities, recommendations, or suggested doctors "
             "mentioned in the reports processed in this session.")
        )
        print("[INFO] Retriever tool created.")

        # Create Agent Executor
        print("[INFO] Setting up chat LLM and Agent Executor...")
        try:
            llm = ChatGroq(model="llama3-70b-8192", temperature=0) # Use a capable Groq model
            # Updated Prompt for clarity and better tool usage
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are 'MediGuide', a helpful AI assistant for the Medicronus system. Your goal is to answer user questions based *primarily* on the provided medical report context.\n"
                    "Use the 'Patient_Records_Context_Tool' to access information from the analyzed report (findings, explanations, abnormalities, recommendations, doctors list).\n"
                    "Answer concisely and clearly. If the answer isn't found in the tool's context, state that the information is not available in the current report analysis.\n"
                    "Do not provide general medical advice beyond what is stated in the retrieved context.\n"
                    "Chat History:\n{chat_history}"
                )),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            agent = create_openai_tools_agent(llm=llm, tools=[tool], prompt=prompt)
            agent_exec = AgentExecutor(
                agent=agent,
                tools=[tool],
                verbose=True, # Set to True for debugging agent steps
                handle_parsing_errors=True # Gracefully handle LLM output issues
            )
            print("[INFO] Chat Agent Executor created successfully.")
            st.session_state.agent_executor = agent_exec # Store in session state
            return agent_exec
        except Exception as e_agent:
            print(f"[CRITICAL ERROR] Failed to create chat agent:")
            log_error(e_agent)
            st.error("Failed to initialize the chat interface agent.")
            return None

    except Exception as e_setup:
        print(f"[CRITICAL ERROR] Unexpected error during chat setup:")
        log_error(e_setup)
        st.error("An unexpected error occurred while setting up the chat interface.")
        return None


# --- Streamlit App Layout ---

st.title("Medicronus MARK I - Medical Report Analysis System")

# Instantiate the logic class
meditrust_ai = MediTrustAI()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Upload Report", "🔍 Analysis Results", "👨‍⚕️ Find Doctors", "💬 Chat"])

# Tab 1: Upload Report
with tab1:
    st.header("Upload Your Medical Report")
    st.markdown("""
    Welcome to Medicronus MARK I. Upload your medical report (PDF format) to begin the analysis.
    Ensure required API keys are configured in the sidebar first.
    """)

    # Check for essential API key for analysis
    if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
        st.error("OpenAI API Key is missing. Please configure it in the sidebar under API Configuration.")
        st.stop() # Stop rendering this tab if key is missing

    uploaded_file = st.file_uploader("Choose a PDF file...", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        # Save the uploaded file temporarily for processing
        if st.session_state.temp_dir:
            temp_file_path = os.path.join(st.session_state.temp_dir, "uploaded_report.pdf")
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                print(f"[INFO] Uploaded file saved to: {temp_file_path}")

                # Read the report content immediately for analysis input
                loader = PyPDFLoader(temp_file_path)
                report_docs = loader.load()
                report_content = "\n\n".join([doc.page_content for doc in report_docs])
                st.session_state.report_content = report_content # Store raw content
                st.success("File uploaded successfully and content extracted.")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Report Preview (first 500 chars):")
                    st.text_area("Raw Content Preview", value=report_content[:500] + "...", height=250, disabled=True, key="report_preview_area")
                with col2:
                    st.subheader("Start Analysis")
                    st.markdown("Click below to start the AI analysis. This may take a few minutes.")
                    if st.button("🔍 Analyze Report", key="analyze_button", type="primary"):
                         if st.session_state.report_content:
                             with st.spinner('Analyzing your medical report... Please wait.'):
                                 analysis_success = meditrust_ai.run_analysis(st.session_state.report_content)
                             if analysis_success:
                                 st.success('Analysis complete! Check the "Analysis Results" tab.')
                                 # Trigger chat setup *after* successful analysis
                                 with st.spinner("Initializing chat interface..."):
                                      setup_chat_interface()
                             else:
                                 st.error('Analysis failed or could not retrieve results. Check console logs for details.')
                         else:
                             st.warning("Report content seems missing. Please re-upload.")

            except Exception as e_upload:
                log_error(e_upload)
                st.error("Failed to save or process the uploaded file.")
        else:
            st.error("Temporary directory not available. Cannot process file.")

# Tab 2: Analysis Results
with tab2:
    st.header("Medical Report Analysis Results")
    if st.session_state.analysis_complete:
        tab2_1, tab2_2, tab2_3 = st.tabs(["📄 Structured Report", "📊 Explained Report", "⚠️ Abnormalities"])

        with tab2_1:
            st.subheader("Structured Report")
            st.markdown(st.session_state.get("extracted_report", "No structured report generated."))
            # Add download button
            st.download_button(
                label="Download Structured Report (Markdown)",
                data=st.session_state.get("extracted_report", ""),
                file_name="structured_report.md",
                mime="text/markdown",
            )

        with tab2_2:
            st.subheader("Explained Report with Normal/Abnormal Classifications")
            st.markdown(st.session_state.get("explained_report", "No explained report generated."))
            st.download_button(
                label="Download Explained Report (Markdown)",
                data=st.session_state.get("explained_report", ""),
                file_name="explained_report.md",
                mime="text/markdown",
            )

        with tab2_3:
            st.subheader("Abnormalities and Recommendations")
            st.markdown(st.session_state.get("abnormalities", "No abnormalities report generated."))
            st.download_button(
                label="Download Abnormalities Report (Markdown)",
                data=st.session_state.get("abnormalities", ""),
                file_name="abnormalities.md",
                mime="text/markdown",
            )
    else:
        st.info("Please upload and successfully analyze a report in the 'Upload Report' tab first.")

# Tab 3: Find Doctors
with tab3:
    st.header("Find Specialists Based on Your Report")

    # Check prerequisites for doctor search
    can_search = False
    if not st.session_state.analysis_complete:
         st.info("Please upload and successfully analyze a report first.")
    elif not st.session_state.abnormalities or "Placeholder content" in st.session_state.abnormalities or "not generated" in st.session_state.abnormalities:
         st.warning("Abnormalities report section is missing or invalid. Cannot search for doctors without it.")
    elif 'SERPER_API_KEY' not in os.environ or not os.environ['SERPER_API_KEY']:
         st.warning("SERPER API Key is missing (configure in sidebar). Doctor search requires this key.")
    else:
         can_search = True

    if can_search:
        st.markdown("Enter your city to find specialists relevant to the detected abnormalities.")
        user_city = st.text_input("Enter your city:", key="city_input")

        if st.button("🔍 Find Doctors", key="find_doctors_button", type="primary"):
            if user_city:
                 with st.spinner('Searching for specialists in your area...'):
                     search_success = meditrust_ai.run_doctor_search(user_city, st.session_state.abnormalities)
                 if search_success:
                     st.success('Doctor search complete! See results below.')
                     # Trigger chat setup/update *after* successful search if not already done
                     if not st.session_state.agent_executor:
                         with st.spinner("Updating chat interface with doctor info..."):
                             setup_chat_interface()
                 else:
                     st.error('Doctor search failed or could not retrieve results. Check console logs.')
            else:
                st.warning("Please enter your city.")

    # Display results if search is complete
    if st.session_state.doctor_search_complete:
        st.subheader("Recommended Specialists")
        st.markdown(st.session_state.get("doctors_list", "No doctor information available."))
        st.download_button(
            label="Download Doctors List (Markdown)",
            data=st.session_state.get("doctors_list", ""),
            file_name="doctors_list.md",
            mime="text/markdown",
        )

# Tab 4: Chat with MediGuide
with tab4:
    st.header("Chat with MediGuide")

    if st.session_state.analysis_complete:
        st.markdown("Ask questions about your analyzed report. MediGuide uses the generated context.")

        # Attempt to setup chat if not already done (e.g., if user navigates here directly after analysis)
        if not st.session_state.agent_executor:
             print("[INFO] Chat agent not found in state, attempting setup...")
             setup_chat_interface()

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["bot"])

        # Chat input
        user_query = st.chat_input("Ask about your report...", key="chat_input")

        if user_query:
            # Add user message to chat history immediately
            st.session_state.chat_history.append({"user": user_query, "bot": ""}) # Add placeholder for bot response
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)

            # Check if agent is ready
            if st.session_state.agent_executor:
                # Display thinking indicator and get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                             response = st.session_state.agent_executor.invoke({
                                 "input": user_query,
                                 # Pass history correctly for Langchain memory placeholders
                                 "chat_history": [(msg["user"], msg["bot"]) for msg in st.session_state.chat_history[:-1]] # Exclude current empty bot msg
                             })
                             bot_response = response.get("output", "Sorry, I couldn't process that.")
                             st.write(bot_response)
                             # Update the last history item with the actual bot response
                             st.session_state.chat_history[-1]["bot"] = bot_response
                        except Exception as e_chat:
                             print(f"[ERROR] Error during agent invocation:")
                             log_error(e_chat)
                             error_msg = f"Sorry, I encountered an error processing your request: {str(e_chat)}"
                             st.error(error_msg)
                             st.session_state.chat_history[-1]["bot"] = error_msg # Update history with error

            else:
                # Handle case where agent failed to initialize
                 error_msg = "Chat interface is not available. Please ensure analysis was successful and all API keys are correctly configured."
                 with st.chat_message("assistant"):
                     st.error(error_msg)
                 st.session_state.chat_history[-1]["bot"] = error_msg # Update history
    else:
        st.info("Please upload and successfully analyze a report first to enable chat.")


# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Medicronus MARK I - Advanced Medical Report Analysis System")

# --- Cleanup (Optional) ---
# Consider adding cleanup for the temp directory if running locally and desired
# import shutil
# def cleanup_temp_dir():
#     if 'temp_dir' in st.session_state and st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
#         try:
#             shutil.rmtree(st.session_state.temp_dir)
#             print(f"[INFO] Cleaned up temp directory: {st.session_state.temp_dir}")
#             del st.session_state.temp_dir
#         except Exception as e:
#             print(f"[WARN] Could not remove temp directory {st.session_state.temp_dir}: {e}")
# # You might register this with atexit or implement a session cleanup mechanism if needed.
