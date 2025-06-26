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
    /* ... (Your existing CSS) ... */
    .main-title { color: #1E88E5; text-align: center; padding-bottom: 10px; font-family: 'Arial', sans-serif; }
    .sidebar-title { color: #1E88E5; font-family: 'Arial', sans-serif; }
    .stSidebar .css-1d391kg { background-color: #f4f6f8; }
    .tutor-card { background-color: #ffffff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .tutor-card img { border-radius: 50%; width: 100px; height: 100px; object-fit: cover; margin-bottom: 12px; border: 3px solid #e0e0e0; }
    .tutor-name { font-size: 1.15em; font-weight: 600; color: #333745; margin-bottom: 5px; }
    .tutor-bio { font-size: 0.85em; color: #505050; line-height: 1.4; }
    .stChatMessage[data-testid="chatAvatarIcon-user"] + div div[data-testid="stMarkdownContainer"] { background-color: #e3f2fd; border-radius: 18px; border-top-left-radius: 5px; padding: 10px 15px; margin-right: 50px; }
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div div[data-testid="stMarkdownContainer"] { background-color: #e8f5e9; border-radius: 18px; border-top-right-radius: 5px; padding: 10px 15px; margin-left: 50px; }
    .stSpinner > div > div { text-align: center; }
    .processed-doc-title { font-size: 0.9em; font-weight: bold; color: #444; margin-top:10px; margin-bottom:5px;}
    .processed-doc-item { font-size: 0.85em; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource(show_spinner=False)
def get_cached_tutor(personality_name_for_cache: str, socratic_mode_for_cache: bool, doc_processing_trigger: int):
    # print(f"Streamlit: Init/cache tutor: P:{personality_name_for_cache}, S:{socratic_mode_for_cache}, D:{doc_processing_trigger}")
    tutor = get_tutor_instance(
        personality_name=personality_name_for_cache,
        socratic_mode=socratic_mode_for_cache
    )
    if not tutor or not tutor.llm:
        return None
    return tutor

# --- Main App Structure ---
available_personalities_dict = get_available_personalities()
all_personality_names = sorted(list(available_personalities_dict.keys()))

# Initialize session state
if "selected_personality" not in st.session_state:
    st.session_state.selected_personality = all_personality_names[0]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "socratic_mode_active" not in st.session_state:
    st.session_state.socratic_mode_active = True
if "tutor_state_changed_flag" not in st.session_state: # Combined flag
    st.session_state.tutor_state_changed_flag = True
if "doc_processing_trigger" not in st.session_state:
    st.session_state.doc_processing_trigger = 0
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []
if "last_uploaded_filenames_set" not in st.session_state:
    st.session_state.last_uploaded_filenames_set = set()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌟 MentorMind Options</h2>", unsafe_allow_html=True)
    st.markdown("Your personal AI Socratic learning companion.")
    st.markdown("---")
    st.markdown("### Choose Your Tutor:")

    current_selection_index = all_personality_names.index(st.session_state.selected_personality)
    newly_selected_personality_name_from_ui = st.selectbox(
        "Tutor Personality:", options=all_personality_names, index=current_selection_index, key="personality_sb"
    )
    
    st.markdown("### Learning Mode:")
    newly_selected_mode = st.toggle(
        "Socratic Guidance Mode", value=st.session_state.socratic_mode_active, key="socratic_toggle",
        help="ON: Guided Socratic learning. OFF: Direct answers."
    )

    st.markdown("---")
    st.markdown("### Upload Document(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF files to discuss:", type=["pdf"], accept_multiple_files=True, key="file_uploader"
    )

    # Get active tutor - this will re-cache if personality, mode, or doc_trigger changes
    active_tutor = get_cached_tutor(
        newly_selected_personality_name_from_ui,
        newly_selected_mode,
        st.session_state.doc_processing_trigger
    )

    # Document Processing Logic
    if uploaded_files and active_tutor:
        current_uploaded_filenames_set = set(f.name for f in uploaded_files)
        if current_uploaded_filenames_set != st.session_state.last_uploaded_filenames_set:
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                file_data_list = [file.getvalue() for file in uploaded_files]
                filenames_list = [file.name for file in uploaded_files]
                if not active_tutor.llm:
                    st.error("Tutor LLM not ready. Cannot process docs.")
                else:
                    success = active_tutor.process_documents(file_data_list, filenames_list)
                    if success:
                        st.session_state.processed_filenames = filenames_list
                        st.session_state.last_uploaded_filenames_set = current_uploaded_filenames_set
                        st.session_state.doc_processing_trigger += 1
                        st.session_state.tutor_state_changed_flag = True
                        st.rerun()
                    else:
                        st.error("Failed to process document(s).")
                        st.session_state.processed_filenames = []
                        st.session_state.last_uploaded_filenames_set = set()
    elif not uploaded_files and st.session_state.last_uploaded_filenames_set: # Files removed
        if active_tutor:
            with st.spinner("Clearing document context..."):
                active_tutor.process_documents([], [])
        st.session_state.processed_filenames = []
        st.session_state.last_uploaded_filenames_set = set()
        st.session_state.doc_processing_trigger += 1
        st.session_state.tutor_state_changed_flag = True
        st.rerun()

    if st.session_state.processed_filenames:
        st.sidebar.markdown("<div class='processed-doc-title'>Active Document(s):</div>", unsafe_allow_html=True)
        for fname in st.session_state.processed_filenames:
            st.sidebar.markdown(f"<span class='processed-doc-item'>📄 {fname}</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Tutor Card Display
    if active_tutor:
        # ... (Your existing tutor card display logic using active_tutor, available_personalities_dict, IMAGE_DIR) ...
        current_personality_key = active_tutor.get_current_personality_name()
        personality_details = available_personalities_dict.get(current_personality_key, {})
        st.markdown("<div class='tutor-card'>", unsafe_allow_html=True)
        image_filename = personality_details.get("image_filename")
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path): st.image(image_path)
            else: st.caption(f"Img: {image_filename} missing")
        st.markdown(f"<div class='tutor-name'>{current_personality_key}</div>", unsafe_allow_html=True)
        bio = personality_details.get("bio", "No bio.")
        st.markdown(f"<div class='tutor-bio'>{bio}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Tutor couldn't initialize.")
    st.markdown("---")

    if st.button("Clear Chat & Docs 🧹", key="clear_btn", use_container_width=True):
        st.session_state.messages = []
        if active_tutor: active_tutor.process_documents([], []) # Clear docs in tutor
        get_cached_tutor.clear() # Clear all cached tutors
        st.session_state.tutor_state_changed_flag = True
        st.session_state.processed_filenames = []
        st.session_state.last_uploaded_filenames_set = set()
        st.session_state.doc_processing_trigger += 1
        st.rerun()

# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>🎓 MentorMind - AI Socratic Tutor</h1>", unsafe_allow_html=True)
mode_display_str = "Socratic Guidance" if st.session_state.socratic_mode_active else "Direct Answers"
model_disp_main = LLM_MODEL_NAME
if active_tutor: model_disp_main = active_tutor.get_llm_model_name()
st.markdown(f"<p style='text-align: center; color: grey; margin-bottom: 20px;'>Interacting as: <b>{st.session_state.selected_personality}</b> | Mode: <b>{mode_display_str}</b> | Model: {model_disp_main}</p>", unsafe_allow_html=True)

# Handle UI state changes (personality or mode)
if st.session_state.selected_personality != newly_selected_personality_name_from_ui or \
   st.session_state.socratic_mode_active != newly_selected_mode:
    st.session_state.selected_personality = newly_selected_personality_name_from_ui
    st.session_state.socratic_mode_active = newly_selected_mode
    st.session_state.messages = [] # Clear displayed messages
    st.session_state.tutor_state_changed_flag = True # Trigger welcome message
    st.rerun()

# Display initial message or existing messages
if not active_tutor or not active_tutor.chain:
    st.error("Tutor is not fully initialized. Please check settings.")
else:
    if st.session_state.tutor_state_changed_flag or not st.session_state.messages:
        mode_intro = "I'll guide you Socratically." if active_tutor.socratic_mode else "I'll provide direct answers."
        doc_intro = f" We can also discuss: {', '.join(st.session_state.processed_filenames)}." if st.session_state.processed_filenames else ""
        initial_message_content = f"Hello! I am {active_tutor.get_current_personality_name()}. {mode_intro}{doc_intro} How can I assist?"
        
        if not st.session_state.messages or st.session_state.messages[0].get("content") != initial_message_content:
            st.session_state.messages.insert(0, {"role": "assistant", "content": initial_message_content})
        st.session_state.tutor_state_changed_flag = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input(f"Ask {st.session_state.selected_personality}..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner(f"{st.session_state.selected_personality} is thinking..."):
                response = active_tutor.get_response_text(user_prompt)
            placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
