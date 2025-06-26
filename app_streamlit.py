# app_streamlit.py
import streamlit as st
import os
from tutor_logic import get_tutor_instance, get_available_personalities, LLM_MODEL_NAME

# --- Configuration for Image Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# --- Page Configuration ---
st.set_page_config(
    page_title="MentorMind - AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* General Page Styles */
    .stApp {
        /* background-color: #f9f9f9; */ /* Optional: very light page background */
    }
    /* Main Title */
    .main-title {
        color: #1E88E5; /* Professional Blue */
        text-align: center;
        padding: 20px 0; /* More padding */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300; /* Lighter font weight for modern feel */
        letter-spacing: 1px;
    }
    /* Sidebar Enhancements */
    .sidebar-title {
        color: #1E88E5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 500;
        margin-top: 0px; /* Align with top */
    }
    .stSidebar { /* Targets sidebar overall container */
        /* background-color: #f0f2f6; */ /* Using Streamlit's default theme is often cleaner */
        padding-top: 1rem; /* Add some padding at the top of sidebar content */
    }
    /* Tutor Card Styling */
    .tutor-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    .tutor-card img {
        border-radius: 50%;
        width: 100px; 
        height: 100px;
        object-fit: cover;
        margin-bottom: 15px;
        border: 4px solid #1E88E5; /* Match title color */
    }
    .tutor-name {
        font-size: 1.2em;
        font-weight: 600;
        color: #2c3e50; /* Darker, more professional blue/grey */
        margin-bottom: 8px;
    }
    .tutor-bio {
        font-size: 0.9em;
        color: #555f6d; /* Softer grey */
        line-height: 1.5;
        max-height: 100px; /* Limit bio height */
        overflow-y: auto; /* Add scroll for long bios */
    }
    /* Chat Message Styling - using Streamlit's defaults with slight override */
    div[data-testid="stChatMessage"] {
        border-radius: 10px; /* Softer corners for chat bubbles */
        margin-bottom: 10px; /* Space between messages */
    }
    /* For more advanced chat, consider streamlit-chat or custom components */

    /* Processed Document List Styling */
    .processed-doc-title {
        font-size: 1em; /* Slightly larger */
        font-weight: 600;
        color: #2c3e50;
        margin-top: 15px;
        margin-bottom: 8px;
    }
    .processed-doc-item {
        font-size: 0.9em;
        color: #505050;
        padding: 3px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; /* Show ... for long filenames */
    }
    .stButton>button { /* Style Streamlit buttons slightly */
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_resource(show_spinner=False) # Spinner handled manually
def get_cached_tutor(personality_name: str, socratic_mode: bool, _doc_trigger: int): # Underscore to indicate it's a trigger
    """Gets a cached SocraticTutor instance. Cache invalidates if args change."""
    # print(f"CACHE: Getting tutor for P:{personality_name}, S:{socratic_mode}, D:{_doc_trigger}")
    tutor = get_tutor_instance(
        personality_name=personality_name,
        socratic_mode=socratic_mode
    )
    if not tutor or not tutor.llm: # Check if LLM itself failed (API key, model name)
        st.error(f"Critical Error: Could not initialize the AI model for {personality_name}. Please check API key and model configuration in tutor_logic.py.")
        return None
    return tutor

# --- Initialize Session State ---
def init_session_state():
    """Initializes session state variables if they don't exist."""
    available_personalities_dict = get_available_personalities()
    all_personality_names = sorted(list(available_personalities_dict.keys()))

    if "selected_personality" not in st.session_state:
        st.session_state.selected_personality = all_personality_names[0]
    if "socratic_mode_active" not in st.session_state:
        st.session_state.socratic_mode_active = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tutor_state_changed_flag" not in st.session_state:
        st.session_state.tutor_state_changed_flag = True
    if "doc_processing_trigger" not in st.session_state: # For RAG cache invalidation
        st.session_state.doc_processing_trigger = 0
    if "processed_filenames" not in st.session_state:
        st.session_state.processed_filenames = []
    if "last_uploaded_filenames_set" not in st.session_state:
        st.session_state.last_uploaded_filenames_set = set()

init_session_state() # Call initialization

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌟 MentorMind Options</h2>", unsafe_allow_html=True)
    st.caption("Your personal AI Socratic learning companion.") # Changed from markdown for consistency
    st.markdown("---")

    available_personalities_dict = get_available_personalities() # Load here for UI
    all_personality_names = sorted(list(available_personalities_dict.keys()))
    
    current_personality_index = all_personality_names.index(st.session_state.selected_personality)
    ui_selected_personality = st.selectbox(
        "Tutor Personality:", options=all_personality_names, index=current_personality_index, key="personality_sb"
    )
    
    ui_selected_mode = st.toggle(
        "Socratic Guidance Mode", value=st.session_state.socratic_mode_active, key="socratic_toggle",
        help="ON: Guided Socratic learning. OFF: Direct answers."
    )

    st.markdown("---")
    st.markdown("### Upload Document(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF files to discuss:", type=["pdf"], accept_multiple_files=True, key="file_uploader"
    )

    # Get active tutor - this will re-cache if personality, mode, or doc_trigger changes
    # The arguments passed here are crucial for @st.cache_resource to work as intended
    active_tutor = get_cached_tutor(
        ui_selected_personality, # Use the current selection from UI
        ui_selected_mode,        # Use the current selection from UI
        st.session_state.doc_processing_trigger
    )

    # Document Processing Logic
    if uploaded_files and active_tutor:
        current_uploaded_filenames_set = set(f.name for f in uploaded_files)
        # Process only if the set of files has actually changed from the last processed set
        if current_uploaded_filenames_set != st.session_state.last_uploaded_filenames_set:
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                file_data_list = [file.getvalue() for file in uploaded_files]
                filenames_list = [file.name for file in uploaded_files]
                
                if not active_tutor.llm: # Double check LLM readiness
                    st.error("Tutor LLM not ready. Cannot process docs.")
                else:
                    success = active_tutor.process_documents(file_data_list, filenames_list)
                    if success:
                        st.session_state.processed_filenames = filenames_list
                        st.session_state.last_uploaded_filenames_set = current_uploaded_filenames_set
                        # IMPORTANT: Increment trigger to signal that the *data source* for the tutor has changed.
                        # This will cause get_cached_tutor to re-evaluate and potentially give a new instance,
                        # which will then build its RAG chain with the new docs.
                        st.session_state.doc_processing_trigger += 1
                        st.session_state.tutor_state_changed_flag = True # For new welcome message
                        st.rerun() # Rerun to reflect new RAG context and clear chat
                    else:
                        st.error("Failed to process document(s). Check console for details.")
                        # If processing fails, revert to a non-RAG state for this tutor instance
                        active_tutor.process_documents([], []) 
                        st.session_state.processed_filenames = []
                        st.session_state.last_uploaded_filenames_set = set()
                        # Might still need a rerun if state changed
                        st.session_state.tutor_state_changed_flag = True
                        st.rerun()

    elif not uploaded_files and st.session_state.last_uploaded_filenames_set: # All files removed by user
        if active_tutor:
            with st.spinner("Clearing document context..."):
                active_tutor.process_documents([], []) # Tell tutor to clear its vector_store
        st.session_state.processed_filenames = []
        st.session_state.last_uploaded_filenames_set = set()
        st.session_state.doc_processing_trigger += 1 # Force re-cache for tutor without docs
        st.session_state.tutor_state_changed_flag = True
        st.rerun()

    if st.session_state.processed_filenames:
        st.sidebar.markdown("<div class='processed-doc-title'>Active Document(s):</div>", unsafe_allow_html=True)
        for fname in st.session_state.processed_filenames:
            st.sidebar.markdown(f"<span class='processed-doc-item'>📄 {fname}</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Tutor Card Display
    if active_tutor:
        current_personality_key_display = active_tutor.get_current_personality_name() # From the active tutor
        personality_details = available_personalities_dict.get(current_personality_key_display, {})
        
        st.markdown("<div class='tutor-card'>", unsafe_allow_html=True)
        image_filename = personality_details.get("image_filename")
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path): st.image(image_path, use_column_width='auto') # CSS handles sizing
            else: st.caption(f"Img: {image_filename} missing")
        st.markdown(f"<div class='tutor-name'>{current_personality_key_display}</div>", unsafe_allow_html=True)
        bio = personality_details.get("bio", "No biography available.")
        st.markdown(f"<div class='tutor-bio'>{bio}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # This case should ideally be caught by get_cached_tutor and show an error there
        st.sidebar.error("Tutor instance is not available.")
    st.markdown("---")

    if st.button("Clear Chat & Docs 🧹", key="clear_btn", use_container_width=True):
        st.session_state.messages = []
        if active_tutor: active_tutor.process_documents([], []) 
        get_cached_tutor.clear() # Clear all cached tutor instances
        st.session_state.tutor_state_changed_flag = True
        st.session_state.processed_filenames = []
        st.session_state.last_uploaded_filenames_set = set()
        st.session_state.doc_processing_trigger += 1 # Force new tutor instance on next run
        st.rerun()

# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>🎓 MentorMind AI Tutor</h1>", unsafe_allow_html=True)

# Check for state changes (personality or mode) that require a refresh
# This needs to happen BEFORE displaying the caption and initializing messages
if st.session_state.selected_personality != ui_selected_personality or \
   st.session_state.socratic_mode_active != ui_selected_mode:
    
    st.session_state.selected_personality = ui_selected_personality
    st.session_state.socratic_mode_active = ui_selected_mode
    st.session_state.messages = [] # Clear displayed messages for new tutor/mode
    st.session_state.tutor_state_changed_flag = True # Trigger welcome message for new setup
    # The change in personality/mode will cause get_cached_tutor to fetch/create a new tutor.
    # If documents are in st.file_uploader, the new tutor instance will process them
    # due to the logic: `elif not active_tutor.vector_store and current_uploaded_filenames_set:`
    # No need to change doc_processing_trigger here, as personality/mode args to cache change.
    st.rerun()

# Display current settings
mode_display_str = "Socratic Guidance" if st.session_state.socratic_mode_active else "Direct Answers"
model_disp_main = LLM_MODEL_NAME # Default if active_tutor is None
if active_tutor: model_disp_main = active_tutor.get_llm_model_name()

# Use st.session_state.selected_personality for consistency after potential rerun
st.markdown(f"<p style='text-align: center; color: grey; margin-bottom: 20px;'>Interacting as: <b>{st.session_state.selected_personality}</b> | Mode: <b>{mode_display_str}</b> | Model: {model_disp_main}</p>", unsafe_allow_html=True)


# Display initial message or existing messages
if not active_tutor or not active_tutor.chain: # Check if chain is also initialized
    st.error("Tutor is not fully initialized. Please check settings (API key, model name) and document processing status.")
else:
    # Initial message logic
    if st.session_state.tutor_state_changed_flag or not st.session_state.messages:
        mode_intro = "I'll guide you Socratically." if active_tutor.socratic_mode else "I'll provide direct answers."
        doc_intro = f" We can also discuss: {', '.join(st.session_state.processed_filenames)}." if st.session_state.processed_filenames else ""
        initial_message_content = f"Hello! I am {active_tutor.get_current_personality_name()}. {mode_intro}{doc_intro} How can I assist?"
        
        # Avoid duplicate initial messages if already present
        if not st.session_state.messages or st.session_state.messages[0].get("content") != initial_message_content:
            st.session_state.messages.insert(0, {"role": "assistant", "content": initial_message_content})
        st.session_state.tutor_state_changed_flag = False # Reset flag after displaying initial message

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_prompt := st.chat_input(f"Ask {st.session_state.selected_personality}..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty() # For the response
            with st.spinner(f"{st.session_state.selected_personality} is thinking..."):
                response = active_tutor.get_response_text(user_prompt)
            placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
