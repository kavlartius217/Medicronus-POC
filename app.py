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
from crewai.flow import Flow, start, listen, and_, or_, router
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

# Apply nest_asyncio for compatibility with Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Medicronus MARK I",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
def log_error(error):
    """Log error to Streamlit and to file if possible"""
    st.error(f"Error: {str(error)}")
    error_trace = traceback.format_exc()
    print(f"ERROR: {str(error)}\n{error_trace}")
    try:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"error_log_{int(time.time())}.txt"), "w") as f:
            f.write(f"Error: {str(error)}\n\n{error_trace}")
    except Exception as e:
        print(f"Failed to write error log: {str(e)}")

# Sidebar for API keys
with st.sidebar:
    st.image("https://i.ibb.co/M8JQs2v/medicronus-logo.png", width=250)
    st.title("Medicronus MARK I")
    st.subheader("Healthcare AI Assistant")
    
    with st.expander("⚙️ API Configuration", expanded=False):
        serper_api = st.text_input("SERPER_DEV_TOOL API Key", type="password")
        openai_api = st.text_input("OpenAI API Key", type="password")
        groq_api = st.text_input("Groq API Key", type="password")
        neo4j_uri = st.text_input("Neo4j URI", value="neo4j+s://5b97a604.databases.neo4j.io")
        neo4j_username = st.text_input("Neo4j Username", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password")
        
        if st.button("Save API Keys"):
            try:
                # Check if keys are provided
                if not serper_api or not openai_api or not groq_api or not neo4j_password:
                    st.warning("Please fill in all required API keys.")
                else:
                    os.environ['SERPER_DEV_TOOL'] = serper_api
                    os.environ['OPENAI_API_KEY'] = openai_api
                    os.environ['GROQ_API_KEY'] = groq_api
                    os.environ['NEO4J_URI'] = neo4j_uri
                    os.environ['NEO4J_USERNAME'] = neo4j_username
                    os.environ['NEO4J_PASSWORD'] = neo4j_password
                    st.success("API keys saved successfully!")
            except Exception as e:
                log_error(e)
                st.error("Failed to save API keys.")

# Create temporary directory to store files
if 'temp_dir' not in st.session_state:
    try:
        st.session_state.temp_dir = tempfile.mkdtemp()
        # Create essential directories
        os.makedirs(os.path.join(st.session_state.temp_dir), exist_ok=True)
        
        # Create placeholder files to avoid FileNotFoundError
        for file_name in ["report.md", "explained_report.md", "abnormalities.md", "doctors.md"]:
            with open(os.path.join(st.session_state.temp_dir, file_name), "w") as f:
                f.write("Placeholder content. This file will be populated during analysis.")
    except Exception as e:
        log_error(e)
        st.error("Failed to create temporary directories. App may not function correctly.")

# Initialize session state variables
if 'report_content' not in st.session_state:
    st.session_state.report_content = None
if 'extracted_report' not in st.session_state:
    st.session_state.extracted_report = None
if 'explained_report' not in st.session_state:
    st.session_state.explained_report = None
if 'abnormalities' not in st.session_state:
    st.session_state.abnormalities = None
if 'doctors_list' not in st.session_state:
    st.session_state.doctors_list = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'doctor_search_complete' not in st.session_state:
    st.session_state.doctor_search_complete = False


# Define the CrewBase classes
@CrewBase
class Medic_Bot():
    """The Medic Bot analyzes the health report and generates comprehensive and useful insights"""

    @agent
    def report_extractor_agent(self) -> Agent:
        return Agent(
            role="Extract, organize, and structure report contents into a clear and well-formatted document.",
            goal="Produce a logically structured and comprehensive report that enhances readability while preserving all information.",
            backstory="You are an expert in report structuring, ensuring that extracted content is well-organized, clearly formatted, and easy to navigate. Your role is to transform raw report data into a polished document with logical flow and coherence.",
            memory=True,
            verbose=True
        )

    @agent
    def report_explanation_agent(self) -> Agent:
        return Agent(
            role="Analyze the report received from the report_extractor_agent, evaluate all values, and determine which fall within the normal range and which deviate.",
            goal="Generate a well-structured and comprehensive report that classifies values as normal or abnormal, explains their significance.",
            backstory="You are an experienced pathologist specializing in interpreting blood reports with deep expertise in analyzing medical values. "
            "Your role is to evaluate, explain, and provide meaningful insights to help understand the report findings effectively.",
            memory=True,
            verbose=True
        )

    @agent
    def abnormalities_agent(self) -> Agent:
        return Agent(
            role="Analyze the report received from the report_extractor, identify abnormal values, and focus on those. "
            "Provide actionable recommendations for improvement through lifestyle changes, diet, or medication where applicable. "
            "Assess the severity of abnormalities and determine if immediate medical consultation is necessary.",
            goal="Generate a structured and detailed report that highlights abnormal values, explains their significance, "
            "and provides clear, practical recommendations for improvement. Flag urgent cases for immediate medical attention.",
            backstory="You are a highly experienced medical analyst specializing in identifying health risks and providing data-driven recommendations. "
            "Your expertise lies in interpreting abnormal health metrics and offering practical guidance on managing or improving them. "
            "You ensure that users understand their health status and take appropriate action when needed.",
            memory=True,
            verbose=True
        )

    @task
    def report_extractor_task(self) -> Task:
        return Task(
            description="Extract and organize the contents of the report {report} into a well-structured, coherent, and comprehensive document. "
            "Ensure logical flow, clarity, and readability by using appropriate headings, subheadings, and formatting.",
            expected_output="A complete and professionally structured report that maintains all essential information while improving organization "
            "and presentation. The final document should be clear, concise, and easy to navigate.",
            output_file=f"{st.session_state.temp_dir}/report.md",
            agent=self.report_extractor_agent()
        )

    @task
    def report_explanation_task(self) -> Task:
        return Task(
            description="Analyze the structured report provided by the report_extractor_agent. "
            "Evaluate all numerical and categorical values to classify them as normal or abnormal. "
            "Assign a severity score to each abnormality and provide a detailed explanation, including potential causes, health implications, and medical recommendations. "
            "Ensure the report is structured for clarity, using headings, subheadings, and bullet points where necessary. ",
            expected_output="A well-organized, detailed medical report that: \n"
            "- Clearly classifies each value as normal or abnormal.\n"
            "- Assigns a severity score to every abnormality.\n"
            "- Provides meaningful explanations for abnormal values, including possible causes and medical implications.\n"
            "- Offers actionable recommendations where applicable.\n"
            "- Is structured with clear headings, subheadings, and bullet points for readability.\n",
            agent=self.report_explanation_agent(),
            output_file=f"{st.session_state.temp_dir}/explained_report.md"
        )

    @task
    def abnormalities_task(self) -> Task:
        return Task(
            description="Analyze the report from the report_explanation_agent, focusing exclusively on abnormal values. "
            "For each abnormal metric, determine its potential health risks, explain why it is outside the normal range, "
            "and provide specific recommendations for improvement. "
            "Offer targeted advice on lifestyle changes, dietary modifications, and possible treatments. "
            "Assess the severity of each abnormality and indicate whether urgent medical consultation is required.",
            expected_output="A structured and detailed abnormalities report that:\n"
            "- Lists all abnormal values along with their medical significance.\n"
            "- Explains why these values are outside the normal range and their health implications.\n"
            "- Provides personalized recommendations, including lifestyle, dietary, and medical interventions.\n"
            "- Assigns a severity score to each abnormality (Low / Moderate / High) with justification.\n"
            "- Recommends the type of medical specialist to consult if necessary.\n"
            "- Flags severe cases that require urgent medical attention.\n"
            "- Is formatted with clear headings, subheadings, and bullet points for improved readability and usability.",
            context=[self.report_explanation_task()],
            agent=self.abnormalities_agent(),
            output_file=f"{st.session_state.temp_dir}/abnormalities.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.report_extractor_agent(), self.report_explanation_agent(), self.abnormalities_agent()],
            tasks=[self.report_extractor_task(), self.report_explanation_task(), self.abnormalities_task()]
        )


@CrewBase
class Doctor_Bot():
    """This bot will find the best doctors for the user"""

    @agent
    def doctor_finder_agent(self) -> Agent:
        return Agent(
            role="Doctor Finder Agent",
            goal="Find the best doctors specializing in the detected abnormalities and provide their contact details.",
            backstory="You are an AI-powered healthcare assistant with access to live search tools. "
            "Based on the patient's abnormal blood report parameters, you search for specialists (hematologists, endocrinologists, cardiologists, etc.) "
            "and recommend the most relevant doctors in the patient's location.",
            tools=[SerperDevTool()],
            verbose=True,
            memory=True
        )

    @task
    def doctor_finder_task(self) -> Task:
        return Task(
            description="Search for top-rated doctors specializing in the following abnormalities: {abnormalities}. \n"
            "Ensure the recommended doctors are located in or near {user_city}. \n"
            "For each specialist, provide:\n"
            "1️⃣ Doctor's Name\n"
            "2️⃣ Specialization (e.g., Endocrinologist, Hematologist, Cardiologist)\n"
            "3️⃣ Clinic/Hospital Name\n"
            "4️⃣ Contact Information (Phone/Email)\n"
            "5️⃣ Consultation Options (In-person or Online)\n"
            "6️⃣ Any relevant patient reviews or ratings (if available).",
            expected_output=
            "A list of specialists, including their names, specialization, clinic details, "
            "contact information, and consultation options.",
            agent=self.doctor_finder_agent(),
            output_file=f"{st.session_state.temp_dir}/doctors.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.doctor_finder_agent()],
            tasks=[self.doctor_finder_task()]
        )


class State(BaseModel):
    abnormalities: str = " "
    decision: str = " "
    report: str = " "


class MediTrustAI(Flow[State]):
    @start()
    def medic_bot(self):
        self.state.report = st.session_state.report_content
        try:
            result = (
                Medic_Bot().crew().kickoff({"report": self.state.report})
            )
            self.state.abnormalities = result.raw
            
            # Read the output files - with safety checks
            report_path = f"{st.session_state.temp_dir}/report.md"
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    st.session_state.extracted_report = f.read()
            else:
                st.session_state.extracted_report = "Report extraction failed. Please try again."
            
            explained_path = f"{st.session_state.temp_dir}/explained_report.md"
            if os.path.exists(explained_path):
                with open(explained_path, "r") as f:
                    st.session_state.explained_report = f.read()
            else:
                st.session_state.explained_report = "Report explanation failed. Please try again."
            
            abnorm_path = f"{st.session_state.temp_dir}/abnormalities.md"
            if os.path.exists(abnorm_path):
                with open(abnorm_path, "r") as f:
                    st.session_state.abnormalities = f.read()
            else:
                st.session_state.abnormalities = "Abnormalities analysis failed. Please try again."
            
            st.session_state.analysis_complete = True
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.session_state.analysis_complete = False

    def doctor_bot(self, user_city):
        try:
            result = (
                Doctor_Bot().crew().kickoff({"user_city": user_city, "abnormalities": self.state.abnormalities})
            )
            
            # Read the doctors file with safety check
            doctors_path = f"{st.session_state.temp_dir}/doctors.md"
            if os.path.exists(doctors_path):
                with open(doctors_path, "r") as f:
                    st.session_state.doctors_list = f.read()
            else:
                st.session_state.doctors_list = "Doctor search failed. Please try again."
            
            st.session_state.doctor_search_complete = True
            return st.session_state.doctors_list
        except Exception as e:
            st.error(f"An error occurred during doctor search: {str(e)}")
            st.session_state.doctor_search_complete = False
            return "Error occurred during doctor search."


# Set up Neo4j and ChatBot once data is available
def setup_chat_interface():
    if not st.session_state.analysis_complete:
        return None
    
    try:
        # Create text files for Neo4j - ensure directory exists
        os.makedirs(st.session_state.temp_dir, exist_ok=True)
        
        # Write abnormalities file
        with open(f"{st.session_state.temp_dir}/abnormalities.md", "w") as f:
            f.write(st.session_state.abnormalities)
        
        # Write doctors file if available
        if st.session_state.doctor_search_complete and st.session_state.doctors_list:
            with open(f"{st.session_state.temp_dir}/doctors.md", "w") as f:
                f.write(st.session_state.doctors_list)
        
        # Write explained report file
        with open(f"{st.session_state.temp_dir}/explained_report.md", "w") as f:
            f.write(st.session_state.explained_report)
        
        # Loading the documents with error handling
        rcts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Load and process abnormalities document
        abnorm_path = f"{st.session_state.temp_dir}/abnormalities.md"
        if os.path.exists(abnorm_path):
            doc_1 = TextLoader(abnorm_path)
            doc_1 = doc_1.load()
            doc_1 = rcts.split_documents(doc_1)
        else:
            doc_1 = []
        
        # Load and process explained report document
        explained_path = f"{st.session_state.temp_dir}/explained_report.md"
        if os.path.exists(explained_path):
            doc_3 = TextLoader(explained_path)
            doc_3 = doc_3.load()
            doc_3 = rcts.split_documents(doc_3)
        else:
            doc_3 = []
        
        doc = doc_1 + doc_3
        
        # Load and process doctors document if available
        if st.session_state.doctor_search_complete:
            doctors_path = f"{st.session_state.temp_dir}/doctors.md"
            if os.path.exists(doctors_path):
                doc_2 = TextLoader(doctors_path)
                doc_2 = doc_2.load()
                doc_2 = rcts.split_documents(doc_2)
                doc = doc + doc_2
    
        if not doc:
            st.warning("No documents available for knowledge base. Chat functionality may be limited.")
            return None
            
        try:
            # Creating the graph db
            embeddings = OpenAIEmbeddings()
            db = Neo4jVector.from_documents(
                embedding=embeddings,
                documents=doc
            )
            
            # Creating the bot
            retriever = db.as_retriever()
            tool = create_retriever_tool(
                retriever,
                "Patient_Records_and_Recommendations_Tool",
                "This tool stores patient information, including medical records, suggested treatments, and recommended doctors for each patient."
            )
        except Exception as e:
            st.error(f"Error setting up knowledge base: {str(e)}")
            return None
    
        try:
            # LLM with error handling
            if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY']:
                st.warning("Groq API Key not set. Chat functionality will not work.")
                return None
                
            llm = ChatGroq(model="qwen-qwq-32b")
            
            # Prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a healthcare assistant. Use the patient's health reports and the provided tool and the chat history {chat_history} to accurately answer their questions."),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_openai_tools_agent(llm=llm, tools=[tool], prompt=prompt)
            agent_exec = AgentExecutor(agent=agent, tools=[tool], verbose=False)
            
            return agent_exec
        except Exception as e:
            st.error(f"Error setting up chat interface: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error in chat interface setup: {str(e)}")
        return None


# Main app layout
st.title("Medicronus MARK I - Medical Report Analysis System")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Upload Report", "🔍 Analysis Results", "👨‍⚕️ Find Doctors", "💬 Chat"])

# Tab 1: Upload Report
with tab1:
    st.header("Upload Your Medical Report")
    st.markdown("""
    Welcome to Medicronus MARK I, your AI-powered medical report analysis system. 
    Upload your medical report in PDF format to get started.
    """)
    
    uploaded_file = st.file_uploader("Choose a PDF file...", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file
        temp_file_path = os.path.join(st.session_state.temp_dir, "uploaded_report.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read the report content
        loader = PyPDFLoader(temp_file_path)
        report = loader.load()
        report_content = "\n\n".join([doc.page_content for doc in report])
        st.session_state.report_content = report_content
        
        st.success("File uploaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Report Preview:")
            st.text_area("Raw Content", value=report_content[:500] + "...", height=300, disabled=True)
        
        with col2:
            st.subheader("Start Analysis")
            st.markdown("""
            Click the button below to start the analysis of your medical report. 
            This will extract, explain, and identify abnormalities in your report.
            """)
            
            analyze_btn = st.button("🔍 Analyze Report", type="primary")
            
            if analyze_btn:
                try:
                    with st.spinner('Analyzing your medical report... This might take a few minutes.'):
                        # Check if API keys are set
                        if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
                            st.error("OpenAI API Key not set. Please set it in the sidebar.")
                            return
                            
                        medicronus = MediTrustAI()
                        medicronus.kickoff()
                        
                    if st.session_state.analysis_complete:
                        st.success('Analysis complete!')
                    else:
                        st.error('Analysis failed. Please check logs for details.')
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state.analysis_complete = False

# Tab 2: Analysis Results
with tab2:
    if st.session_state.analysis_complete:
        st.header("Medical Report Analysis Results")
        
        tab2_1, tab2_2, tab2_3 = st.tabs(["📄 Structured Report", "📊 Explained Report", "⚠️ Abnormalities"])
        
        with tab2_1:
            st.subheader("Structured Report")
            st.markdown(st.session_state.extracted_report)
        
        with tab2_2:
            st.subheader("Explained Report with Normal/Abnormal Classifications")
            st.markdown(st.session_state.explained_report)
        
        with tab2_3:
            st.subheader("Abnormalities and Recommendations")
            st.markdown(st.session_state.abnormalities)
    else:
        st.info("Please upload and analyze a report in the 'Upload Report' tab first.")

# Tab 3: Find Doctors
with tab3:
    st.header("Find Specialists Based on Your Report")
    
    if st.session_state.analysis_complete:
        st.markdown("""
        Let us find the best doctors specializing in the abnormalities detected in your report.
        Just enter your city, and we'll search for suitable specialists near you.
        """)
        
        user_city = st.text_input("Enter your city:")
        
        if st.button("🔍 Find Doctors", type="primary"):
            if user_city:
                try:
                    with st.spinner('Searching for specialists in your area... This might take a few minutes.'):
                        # Check if API keys are set
                        if 'SERPER_DEV_TOOL' not in os.environ or not os.environ['SERPER_DEV_TOOL']:
                            st.error("SERPER DEV API Key not set. Please set it in the sidebar.")
                            return
                            
                        medicronus = MediTrustAI()
                        medicronus.state.abnormalities = st.session_state.abnormalities
                        doctors_list = medicronus.doctor_bot(user_city)
                    
                    if st.session_state.doctor_search_complete:
                        st.success('Doctor search complete!')
                    else:
                        st.error('Doctor search failed. Please check logs for details.')
                except Exception as e:
                    st.error(f"Error during doctor search: {str(e)}")
                    st.session_state.doctor_search_complete = False
            else:
                st.warning("Please enter your city first.")
        
        if st.session_state.doctor_search_complete:
            st.subheader("Recommended Specialists")
            st.markdown(st.session_state.doctors_list)
    else:
        st.info("Please upload and analyze a report in the 'Upload Report' tab first.")

# Tab 4: Chat with MediGuide
with tab4:
    st.header("Chat with MediGuide")
    
    if st.session_state.analysis_complete:
        st.markdown("""
        Ask questions about your medical report and get personalized answers from MediGuide, 
        your AI medical assistant.
        """)
        
        # Initialize the chat interface if needed
        agent_exec = setup_chat_interface()
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["bot"])
        
        # Chat input
        user_query = st.chat_input("Ask about your report...")
        
        if user_query:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_query)
            
            if agent_exec is not None:
                try:
                    # Get response
                    with st.spinner("Thinking..."):
                        response = agent_exec.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                        bot_response = response["output"]
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(bot_response)
                    
                    # Update chat history
                    st.session_state.chat_history.append({"user": user_query, "bot": bot_response})
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.error(f"Sorry, I encountered an error: {str(e)}")
                    st.session_state.chat_history.append({"user": user_query, "bot": f"Error: {str(e)}"})
            else:
                with st.chat_message("assistant"):
                    st.error("Chat interface could not be initialized. Please check API keys and try again.")
                st.session_state.chat_history.append({"user": user_query, "bot": "Chat interface initialization error"})
    else:
        st.info("Please upload and analyze a report in the 'Upload Report' tab first.")

# Add footer
st.markdown("---")
st.markdown("© 2025 Medicronus MARK I - Advanced Medical Report Analysis System")
