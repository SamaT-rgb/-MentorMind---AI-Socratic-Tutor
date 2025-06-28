# app_streamlit.py
import streamlit as st
import os
# Ensure LLM_MODEL_NAME is imported for the fallback caption
from tutor_logic import get_tutor_instance, get_available_personalities, LLM_MODEL_NAME

# --- Configuration for Image Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="MentorMind - AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polished UI ---
st.markdown("""
<style>
    /* General Page Styles & Font */
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Main Title Styling */
    .main-title {
        color: #1E88E5; /* A strong, professional blue */
        text-align: center;
        padding-bottom: 10px;
        font-weight: 300; /* Lighter font for a modern feel */
        letter-spacing: 1px;
    }
    /* Sidebar Enhancements */
    .sidebar-title {
        color: #1E88E5;
        font-weight: 500;
        margin-top: 0px;
    }
    .stSidebar {
        border-right: 1px solid #e6e6e6; /* Subtle border */
    }
    /* Tutor Card for the Sidebar */
    .tutor-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .tutor-card img {
        border-radius: 50%; /* Circular images */
        width: 100px; 
        height: 100px;
        object-fit: cover; /* Prevents image stretching */
        margin-bottom: 15px;
        border: 4px solid #1E88E5; /* Themed border */
    }
    .tutor-name {
        font-size: 1.2em;
        font-weight: 600;
        color: #2c3e50; /* Darker, professional text color */
        margin-bottom: 8px;
    }
    .tutor-bio {
        font-size: 0.9em;
        color: #555f6d; /* Softer grey for bio text */
        line-height: 1.4;
        max-height: 100px;
        overflow-y: auto; /* Adds scroll for long bios */
    }
    /* Chat Bubble Styling */
    div[data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #e3f2fd; /* Light blue for user */
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f1f8e9; /* Light green for assistant */
    }
    /* Processed Document List Styling */
    .processed-doc-title { font-size: 1em; font-weight: 600; color: #2c3e50; margin-top: 15px; margin-bottom: 8px; }
    .processed-doc-item { font-size: 0.9em; color: #505050; padding: 3px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Caching ---
@st.cache_resource(show_spinner="Initializing tutor...") # Show a default spinner
def get_cached_tutor(personality_name: str, socratic_mode: bool, _doc_trigger: int):
    """
    Gets a cached SocraticTutor instance. Cache invalidates if any argument changes.
    The underscore in _doc_trigger signals it's an internal trigger, not user-facing data.
    """
    tutor = get_tutor_instance(
        personality_name=personality_name,
        socratic_mode=socratic_mode
    )
    if not tutor or not tutor.llm:
        st.error(f"Critical Error: Could not initialize AI model. Please check API key/model name in your setup.")
        return None
    return tutor

# --- Initialize Session State ---
def init_session_state():
    """Initializes all necessary session state variables."""
    if "selected_personality" not in st.session_state:
        st.session_state.selected_personality = sorted(list(get_available_personalities().keys()))[0]
    if "socratic_mode_active" not in st.session_state:
        st.session_state.socratic_mode_active = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tutor_state_changed_flag" not in st.session_state:
        st.session_state.tutor_state_changed_flag = True
    if "doc_processing_trigger" not in st.session_state:
        st.session_state.doc_processing_trigger = 0
    if "processed_filenames" not in st.session_state:
        st.session_state.processed_filenames = []
    if "last_uploaded_filenames_set" not in st.session_state:
        st.session_state.last_uploaded_filenames_set = set()

init_session_state()

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌟 MentorMind Options</h2>", unsafe_allow_html=True)
    st.caption("Your personal AI Socratic learning companion.")
    st.markdown("---")

    # 1. Personality Selection
    available_personalities_dict = get_available_personalities()
    all_personality_names = sorted(list(available_personalities_dict.keys()))
    current_personality_index = all_personality_names.index(st.session_state.selected_personality)
    ui_selected_personality = st.selectbox(
        "Choose Your Tutor:",
        options=all_personality_names,
        index=current_personality_index,
        help="Select a personality to guide your learning."
    )
    
    # 2. Learning Mode Toggle
    ui_selected_mode = st.toggle(
        "Socratic Guidance Mode",
        value=st.session_state.socratic_mode_active,
        help="ON: Guided Socratic learning. OFF: Direct answers."
    )
    st.markdown("---")

    # 3. File Uploader
    st.markdown("### Upload Document(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF files to discuss:",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # --- Get the active tutor instance based on ALL current UI selections ---
    active_tutor = get_cached_tutor(
        ui_selected_personality,
        ui_selected_mode,
        st.session_state.doc_processing_trigger
    )

    # 4. Tutor Display Card
    if active_tutor:
        personality_details = available_personalities_dict.get(active_tutor.get_current_personality_name(), {})
        
        st.markdown("<div class='tutor-card'>", unsafe_allow_html=True)
        image_filename = personality_details.get("image_filename")
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path):
                # FIX: Replaced use_column_width with use_container_width
                st.image(image_path, use_container_width=True) # CSS handles sizing and shape
            else:
                st.caption(f"Img not found: {image_filename}")
        
        st.markdown(f"<div class='tutor-name'>{active_tutor.get_current_personality_name()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tutor-bio'>{personality_details.get('bio', 'No bio available.')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # 5. Clear Chat Button
    if st.button("Clear Chat & Docs 🧹", use_container_width=True):
        st.session_state.messages = []
        if active_tutor:
            active_tutor.process_documents([], []) # Clear doc context in tutor
        get_cached_tutor.clear()
        st.session_state.tutor_state_changed_flag = True
        st.session_state.processed_filenames = []
        st.session_state.last_uploaded_filenames_set = set()
        st.session_state.doc_processing_trigger += 1 # Force new tutor on next run
        st.rerun()

# --- Document Processing Logic (handles new uploads and removals) ---
if active_tutor:
    current_uploaded_filenames_set = set(f.name for f in uploaded_files)
    if current_uploaded_filenames_set != st.session_state.last_uploaded_filenames_set:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            if not uploaded_files: # If files were removed
                active_tutor.process_documents([], [])
                st.toast("Document context cleared!")
            else:
                file_data_list = [file.getvalue() for file in uploaded_files]
                filenames_list = [file.name for file in uploaded_files]
                if active_tutor.process_documents(file_data_list, filenames_list):
                    st.toast(f"Successfully processed {len(filenames_list)} document(s)!", icon="✅")
                else:
                    st.error("Failed to process document(s). See console for details.")
            
            st.session_state.processed_filenames = [f.name for f in uploaded_files]
            st.session_state.last_uploaded_filenames_set = current_uploaded_filenames_set
            st.session_state.doc_processing_trigger += 1
            st.session_state.tutor_state_changed_flag = True
            st.rerun()

# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>MentorMind AI Tutor</h1>", unsafe_allow_html=True)

# State Change Detection (Personality or Mode)
if st.session_state.selected_personality != ui_selected_personality or st.session_state.socratic_mode_active != ui_selected_mode:
    st.session_state.selected_personality = ui_selected_personality
    st.session_state.socratic_mode_active = ui_selected_mode
    st.session_state.messages = []
    st.session_state.tutor_state_changed_flag = True
    st.rerun()

# Display current status
mode_display_str = "Socratic Guidance" if st.session_state.socratic_mode_active else "Direct Answers"
model_disp_main = active_tutor.get_llm_model_name() if active_tutor else LLM_MODEL_NAME
st.markdown(f"<p style='text-align: center; color: grey; margin-bottom: 20px;'>Interacting as: <b>{st.session_state.selected_personality}</b> | Mode: <b>{mode_display_str}</b></p>", unsafe_allow_html=True)


# --- Chat Interface Logic ---
if not active_tutor or not active_tutor.chain:
    st.warning("Tutor is not fully initialized. Please check API key settings.")
else:
    # Initial welcome message
    if st.session_state.tutor_state_changed_flag or not st.session_state.messages:
        mode_intro = "I'll guide you Socratically." if active_tutor.socratic_mode else "I'll provide direct answers."
        doc_intro = f" We can also discuss: {', '.join(st.session_state.processed_filenames)}." if st.session_state.processed_filenames else ""
        initial_message = f"Hello! I am {active_tutor.get_current_personality_name()}. {mode_intro}{doc_intro} How can I assist?"
        
        st.session_state.messages = [{"role": "assistant", "content": initial_message}]
        st.session_state.tutor_state_changed_flag = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if user_prompt := st.chat_input(f"Ask {st.session_state.selected_personality}..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner(f"{st.session_state.selected_personality} is thinking..."):
                response = active_tutor.get_response_text(user_prompt)
            placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
