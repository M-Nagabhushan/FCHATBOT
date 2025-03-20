import streamlit as st
import fitz
import pandas as pd
from docx import Document as Dip
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pyttsx3
import speech_recognition as sr
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import time
import os
import base64
from langchain.embeddings import HuggingFaceEmbeddings

# Set page configuration
st.set_page_config(page_title="File_analyzer", layout="wide")

# Initialize session state variables if they don't exist
if 'general_chat_history' not in st.session_state:
    st.session_state.general_chat_history = {}
if 'file_chat_history' not in st.session_state:
    st.session_state.file_chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'chat_type' not in st.session_state:
    st.session_state.chat_type = "general"
if 'current_chat_name' not in st.session_state:
    st.session_state.current_chat_name = "New Chat"
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'file_data' not in st.session_state:
    st.session_state.file_data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to generate a unique chat ID
def generate_chat_id():
    return str(int(time.time()))# Creating unique id

# Function to read different file types
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()

def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = Dip(file)  #The document module from the docx lib if used to open, modify, read the document content
    return "\n".join(para.text for para in doc.paragraphs)

def read_csv(file):
    return pd.read_csv(file)

def read_excel(file):
    return pd.read_excel(file)

def read_json(file):
    return pd.read_json(file)#To extract data frame the file should be a path or a file kind of object
# '''
# The type of the file object (which we pass as an input to extract content) is,
# streamlit.runtime.uploaded_file_manager which is a file type object that could
# implement methods like .seek(), .read()---file is not just a normal path
# '''

# Function to extract speech input
def extract_from_voice():
    st.write("Listening... (Speak now)")
    status_placeholder = st.empty()
    status_placeholder.info("Listening...")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2#The intake of voice content stops if there is 2 seconds of silence

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)         # Filters the background noises which have less threshold energy
            audio_text = recognizer.listen(source, timeout=None)# Timeout None indicates it indefinitely takes
            status_placeholder.info("Processing speech...")
            vmsg = recognizer.recognize_google(audio_text)      # used to convert spoken audio into text using Google’s Speech-to-Text API.
            status_placeholder.success(f"You said: {vmsg}")
            return vmsg
    except sr.UnknownValueError:
        status_placeholder.error("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        status_placeholder.error("Could not request results from Google Speech Recognition service.")
        return None
    except Exception as e:
        status_placeholder.error(f"Error: {str(e)}")
        return None


# Function to process text-based files and create a vector database
def process_text_data(data, llm):
    with st.spinner("Processing text data..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
        chunks = splitter.split_text(data)
        documents = [Document(page_content=chunk) for chunk in chunks]#Wrapping Text Chunks as Document Objects
        #This document is from docstore it is used to add metadata to the normal text
        # Show progress
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.001)
            progress_bar.progress(i + 1)

        vector_db = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"))
        #vector_db = FAISS.from_documents(documents, OllamaEmbeddings(model_name="all-minilm"))#this needs ollama

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI language model assistant. Your task is to generate five different versions of the given 
            user question to retrieve relevant documents from a vector database. Your goal is to help the user by 
            providing alternate questions to solve the problem of distance-based similarity search.
            Original Question: {question}
            """
        )
        retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)

        prompt_template = ChatPromptTemplate.from_template("""
            content:
            {context}
            Question: {question}
            """)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()} |
                prompt_template |
                llm |
                StrOutputParser()
        )
        # RAG - retrieval argument generation chain which is using pipeline operator(|) for sequential execution flow
        return chain


# Function to initialize the chatbot
def initialize_bot():
    memory = ConversationBufferMemory()
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
            You are an AI assistant created by Mr Nagabhushan Reddy.Give me perfect answer.
            Conversation history:
            {history}
            User: {input}
            AI:
            """
    )
    # Initialize conversation chain
    conversation = ConversationChain(llm=st.session_state.llm, memory=memory, prompt=prompt)
    return conversation


# Function to save chat history
def save_chat(chat_name, chat_id, chat_type):
    if chat_type == "general":
        st.session_state.general_chat_history[chat_id] = {
            "name": chat_name,
            "messages": st.session_state.messages
        }
    else:
        st.session_state.file_chat_history[chat_id] = {
            "name": chat_name,
            "messages": st.session_state.messages,#The messages that are present in the current chat and stored in this dictionary
            "file_type": st.session_state.file_type #The file type object is stored so there will be no need to reload the file if we want to continue the chat
        }

# Function to load chat history
def load_chat(chat_id, chat_type):
    if chat_type == "general":
        if chat_id in st.session_state.general_chat_history:
            st.session_state.messages = st.session_state.general_chat_history[chat_id]["messages"]
            st.session_state.current_chat_name = st.session_state.general_chat_history[chat_id]["name"]
    else:
        if chat_id in st.session_state.file_chat_history:
            st.session_state.messages = st.session_state.file_chat_history[chat_id]["messages"]
            st.session_state.current_chat_name = st.session_state.file_chat_history[chat_id]["name"]
            st.session_state.file_type = st.session_state.file_chat_history[chat_id]["file_type"]
            # The file_type object which is created before, now we are using it ,and we are loading this all to the current state variables


# Function to create a new chat
def new_chat():
    st.session_state.current_chat_id = generate_chat_id()# Unique id created to store chats using generate chat id
    st.session_state.messages = []
    st.session_state.current_chat_name = "New Chat"


# Function to export chat as a text file
def export_chat():
    if not st.session_state.messages:
        st.warning("No chat to export.")
        return

    chat_content = f"# {st.session_state.current_chat_name}\n\n"
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "AI"
        chat_content += f"**{role}**: {message['content']}\n"

    # Create a download link
    b64 = base64.b64encode(chat_content.encode()).decode()
    filename = f"{st.session_state.current_chat_name.replace(' ', '_')}.txt"
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Chat</a>'#URI(uniform resource identifier)
    # """
    # href="data:file/txt;base64,{b64}" - This is the URL that the link points to:
    #         data: is a special URI scheme that allows embedding data directly in a document
    #         file/txt specifies the MIME type (plain text file)
    #         base64 indicates the encoding method used
    #         {b64} is where the actual base64-encoded content is inserted
    # """
    return href# Returns a html href element which allows us to download the chat


# Initialize LLM
def initialize_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API")
    genai.configure(api_key=api_key)
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# Function to delete a chat from history
def delete_chat(chat_id, chat_type):
    if chat_type == "general":
        if chat_id in st.session_state.general_chat_history:
            del st.session_state.general_chat_history[chat_id]
            # If we're deleting the currently active chat, reset the session
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.session_state.current_chat_name = "New Chat"
    else:
        if chat_id in st.session_state.file_chat_history:
            del st.session_state.file_chat_history[chat_id]
            # If we're deleting the currently active chat, reset the session
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.session_state.current_chat_name = "New Chat"
                st.session_state.file_data = None
                st.session_state.file_type = None
def clear_chat():
    st.session_state.current_chat_id = None
    st.session_state.messages = []
    st.session_state.current_chat_name = "New Chat"
    st.session_state.file_data = None
    st.session_state.file_type = None
# Main app
def main():
    # Set title
    if st.session_state.chat_type=="general":
        st.title("General_Chat")
    else:
        st.title("File_Analyzer")

    # Initialize LLM if not already initialized
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm()

    # Sidebar
    with st.sidebar:
        st.header("Options")

        # Chat type selection
        previous_chat_type = st.session_state.chat_type
        selected_chat_type = st.radio("Select Chat Type", ["General Chat", "File Chat"],
                                      index=0 if st.session_state.chat_type == "general" else 1,
                                      key="chat_type_selector")

        # Convert the selection to internal format
        current_chat_type = "general" if selected_chat_type == "General Chat" else "file"

        # Only update and reset if the chat type has actually changed
        if previous_chat_type != current_chat_type:
            st.session_state.chat_type = current_chat_type
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.session_state.current_chat_name = "New Chat"

            # Reset specific data based on chat type
            if current_chat_type == "file":
                st.session_state.conversation = None
            else:
                st.session_state.file_data = None
                st.session_state.file_type = None
                st.session_state.chain = None

            # Using rerun here to apply the changes immediately
            st.rerun()

        # New chat button
        if st.button("New Chat"):
            #st.session_state.file_type = None
            new_chat()
        if st.button("Clear Chat"):
            clear_chat()
            st.rerun()
        # Chat history
        st.subheader("Chat History")

        # Display general chat history
        if st.session_state.chat_type == "general":
            for chat_id, chat_info in st.session_state.general_chat_history.items():
                cols = st.columns([4, 1])
                with cols[0]:
                    if st.button(f"📝 {chat_info['name']}", key=f"general_{chat_id}"):
                        st.session_state.current_chat_id = chat_id
                        load_chat(chat_id, "general")
                        st.rerun()
                with cols[1]:
                    if st.button("❌", key=f"delete_general_{chat_id}"):
                        delete_chat(chat_id, "general")
                        st.rerun()

        # Display file chat history
        else:
            for chat_id, chat_info in st.session_state.file_chat_history.items():
                cols = st.columns([4.5, 1])
                with cols[0]:
                    if st.button(f"📄 {chat_info['name']}", key=f"file_{chat_id}"):
                        st.session_state.current_chat_id = chat_id
                        load_chat(chat_id, "file")
                        st.rerun()
                with cols[1]:
                    if st.button("❌", key=f"delete_file_{chat_id}"):
                        delete_chat(chat_id, "file")
                        st.rerun()

    # Export chat option in top right
    col1, col2 = st.columns([4.5, 1])
    with col2:
        if st.button("Export Chat"):
            href = export_chat()
            if href:
                st.markdown(href, unsafe_allow_html=True)

    # Chat name input
    with col1:
        st.session_state.current_chat_name = st.text_input("Chat Name", value=st.session_state.current_chat_name)

    # Handle file upload for file chat
    if st.session_state.chat_type == "file":
        file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "csv", "xlsx", "json"])

        if file:
            file_type = file.name.split(".")[-1]
            st.session_state.file_type = file_type

            # Process the file based on its type
            file_readers = {
                "csv": read_csv,
                "xlsx": read_excel,
                "json": read_json,
                "pdf": read_pdf,
                "txt": read_txt,
                "docx": read_docx,
            }

            if file_type in file_readers:
                try:
                    st.session_state.file_data = file_readers[file_type](file)
                    st.success(f"File processed successfully: {file.name}")

                    # Create the chain for text-based files
                    if file_type in ["pdf", "txt", "docx"]:
                        st.session_state.chain = process_text_data(str(st.session_state.file_data),
                                                                   st.session_state.llm)

                    # Create a new chat ID for this file
                    if st.session_state.current_chat_id is None:
                        st.session_state.current_chat_id = generate_chat_id()

                    # Save the file name as chat name if it's a new chat
                    if st.session_state.current_chat_name == "New Chat":
                        st.session_state.current_chat_name = file.name
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            else:
                st.error("Unsupported file type")

    # Voice assistance toggle
    voice_mode_current = st.checkbox("Enable Voice Assistant",
                                     value=st.session_state.voice_mode,
                                     key="voice_mode_selector")

    # Check if voice mode setting has changed
    if voice_mode_current != st.session_state.voice_mode:
        st.session_state.voice_mode = voice_mode_current
        st.rerun()

    # Initialize conversation for general chat if needed
    if st.session_state.chat_type == "general" and st.session_state.conversation is None:
        st.session_state.conversation = initialize_bot()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if st.session_state.voice_mode:
        if st.button("Start Voice Input"):
            user_input = extract_from_voice()
            if user_input:
                process_input(user_input)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_input(user_input)

    # Save chat if there are messages
    if st.session_state.messages and st.session_state.current_chat_id:
        save_chat(st.session_state.current_chat_name, st.session_state.current_chat_id, st.session_state.chat_type)


def process_input(user_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response based on chat type
    with st.spinner("Thinking..."):
        if st.session_state.chat_type == "general":
            response = st.session_state.conversation.run(user_input)
        else:
            if st.session_state.file_data is not None:
                if st.session_state.file_type in ["pdf", "txt", "docx"]:
                    response = st.session_state.chain.invoke(user_input)
                else:
                    response = st.session_state.llm.invoke(
                        f"This is the question: {user_input}\nYou have content: {st.session_state.file_data.to_string()}"
                    ).content
            else:
                response = "Please upload a file first."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)

        # Speak the response if voice mode is enabled
        if st.session_state.voice_mode:
            speak(response)

# Function to speak text
def speak(text):
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:  # Delete =False persists to continue even if some blockage occurs
            temp_filename = f.name                                           # A temporary .wav file is created to store the generated speech.
                                                                             # temp_filename stores the path to this temporary file.
        # Use pyttsx3 to convert text to speech
        engine = pyttsx3.init()                                   #Initializes a pyttsx3 speech engine.
        engine.setProperty('rate', 185)               #Sets the speech rate to 185 words per minute.
        engine.save_to_file(text, temp_filename)                  #Converts text to speech and saves it as a .wav file at temp_filename.
        engine.runAndWait()                                       #runAndWait() ensures the speech synthesis completes before moving forward.

        # Play the audio using streamlit audio
        audio_file = open(temp_filename, 'rb')# The temporary audio file is opened in binary read mode ('rb').
        audio_bytes = audio_file.read()       # The content is read into audio_bytes
        st.audio(audio_bytes, format='audio/wav') # plays audio in streamlit

        # Clean up the temporary file
        os.unlink(temp_filename) # Deleting the temporary file
    except Exception as e:
        if "[WinError 32]" in str(e):
            st.success("To listen audio \"click\"")
        else:
            st.write(str(e))
            st.error(f"Error during text-to-speech: {str(e)}")

# Add missing imports
import tempfile# this allows us to create a temp file which gets destroyed once we close the program

if __name__ == "__main__":
    main()