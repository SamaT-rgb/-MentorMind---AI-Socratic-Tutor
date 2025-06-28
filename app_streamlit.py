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

# --- Custom CSS for Polished UI ---
st.markdown("""
<style>
    .main-title { color: #1E88E5; text-align: center; padding: 20px 0; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    .sidebar-title { color: #1E88E5; font-family: 'Helvetica Neue', sans-serif; font-weight: 500; }
    .tutor-card { background-color: #ffffff; border-radius: 10px; padding: 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center; }
    .tutor-card img { border-radius: 50%; width: 100px; height: 100px; object-fit: cover; margin-bottom: 15px; border: 4px solid #1E88E5; }
    .tutor-name { font-size: 1.2em; font-weight: 600; color: #2c3e50; margin-bottom: 8px; }
    .tutor-bio { font-size: 0.9em; color: #555f6d; line-height: 1.4; max-height: 100px; overflow-y: auto; }
    div[data-testid="stChatMessage"] { border-radius: 10px; margin-bottom: 10px; border: 1px solid #e0e0e0; }
    .processed-doc-title { font-size: 1em; font-weight: 600; color: #2c3e50; margin-top: 15px; margin-bottom: 8px; }
    .processed-doc-item { font-size: 0.9em; color: #505050; padding: 3px 0; }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Caching ---
@st.cache_resource(show_spinner="Initializing tutor...")
def get_cached_tutor(personality_name: str, socratic_mode: bool, _doc_trigger: int):
    tutor = get_tutor_instance(personality_name=personality_name, socratic_mode=socratic_mode)
    if not tutor or not tutor.llm:
        st.error("Critical Error: Could not initialize AI model. Check API key and model configuration.")
        return None
    return tutor

# --- Initialize Session State ---
def init_session_state():
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
    st.caption("Your personal AI learning companion.")
    st.markdown("---")

    available_personalities_dict = get_available_personalities()
    all_personality_names = sorted(list(available_personalities_dict.keys()))
    
    current_personality_index = all_personality_names.index(st.session_state.selected_personality)
    ui_selected_personality = st.selectbox(
        "Choose Your Tutor:", options=all_personality_names, index=current_personality_index, key="personality_sb"
    )
    
    ui_selected_mode = st.toggle(
        "Socratic Guidance Mode", value=st.session_state.socratic_mode_active, key="socratic_toggle",
        help="ON: Guided Socratic learning. OFF: Direct answers."
    )
    st.markdown("---")

    st.markdown("### Upload Document(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF, PPTX, DOCX, or TXT files:",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True
    )
    
    active_tutor = get_cached_tutor(
        ui_selected_personality, ui_selected_mode, st.session_state.doc_processing_trigger
    )

    if uploaded_files and active_tutor:
        current_uploaded_filenames_set = set(f.name for f in uploaded_files)
        if current_uploaded_filenames_set != st.session_state.last_uploaded_filenames_set:
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                success = active_tutor.process_documents(
                    [file.getvalue() for file in uploaded_files], [file.name for file in uploaded_files]
                )
                if success: st.toast(f"Successfully processed {len(uploaded_files)} document(s)!", icon="✅")
                else: st.error("Failed to process document(s).")
                
                st.session_state.processed_filenames = [f.name for f in uploaded_files] if success else []
                st.session_state.last_uploaded_filenames_set = current_uploaded_filenames_set if success else set()
                st.session_state.doc_processing_trigger += 1
                st.session_state.tutor_state_changed_flag = True
                st.rerun()

    elif not uploaded_files and st.session_state.last_uploaded_filenames_set:
        with st.spinner("Clearing document context..."):
            if active_tutor: active_tutor.process_documents([], [])
            st.toast("Document context cleared!")
        st.session_state.processed_filenames, st.session_state.last_uploaded_filenames_set = [], set()
        st.session_state.doc_processing_trigger += 1
        st.session_state.tutor_state_changed_flag = True
        st.rerun()

    if st.session_state.processed_filenames:
        st.sidebar.markdown("<div class='processed-doc-title'>Active Document(s):</div>", unsafe_allow_html=True)
        for fname in st.session_state.processed_filenames:
            st.sidebar.markdown(f"<span class='processed-doc-item'>📄 {fname}</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    if active_tutor:
        personality_details = available_personalities_dict.get(active_tutor.get_current_personality_name(), {})
        st.markdown("<div class='tutor-card'>", unsafe_allow_html=True)
        image_filename = personality_details.get("image_filename")
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path): st.image(image_path, use_container_width=True)
            else: st.caption(f"Img: {image_filename} missing")
        st.markdown(f"<div class='tutor-name'>{active_tutor.get_current_personality_name()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tutor-bio'>{personality_details.get('bio', '')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("Clear Chat & Docs 🧹", use_container_width=True):
        st.session_state.messages = []
        if active_tutor: active_tutor.process_documents([], [])
        get_cached_tutor.clear()
        st.session_state.tutor_state_changed_flag = True
        st.session_state.processed_filenames, st.session_state.last_uploaded_filenames_set = [], set()
        st.session_state.doc_processing_trigger += 1
        st.rerun()

# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>MentorMind AI Tutor</h1>", unsafe_allow_html=True)

if st.session_state.selected_personality != ui_selected_personality or st.session_state.socratic_mode_active != ui_selected_mode:
    st.session_state.selected_personality = ui_selected_personality
    st.session_state.socratic_mode_active = ui_selected_mode
    st.session_state.messages = []
    st.session_state.tutor_state_changed_flag = True
    st.rerun()

mode_display_str = "Socratic Guidance" if st.session_state.socratic_mode_active else "Direct Answers"
model_disp_main = active_tutor.get_llm_model_name() if active_tutor else LLM_MODEL_NAME
st.markdown(f"<p style='text-align: center; color: grey; margin-bottom: 20px;'>Interacting as: <b>{st.session_state.selected_personality}</b> | Mode: <b>{mode_display_str}</b></p>", unsafe_allow_html=True)

if not active_tutor or not active_tutor.chain:
    st.error("Tutor is not fully initialized. Please check terminal for API key errors or other issues.")
else:
    if st.session_state.tutor_state_changed_flag or not st.session_state.messages:
        mode_intro = "I'll guide you Socratically." if active_tutor.socratic_mode else "I'll provide direct answers."
        doc_intro = f" We can also discuss: {', '.join(st.session_state.processed_filenames)}." if st.session_state.processed_filenames else ""
        initial_message_content = f"Hello! I am {active_tutor.get_current_personality_name()}. {mode_intro}{doc_intro} How can I assist?"
        
        st.session_state.messages = [{"role": "assistant", "content": initial_message_content}]
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
