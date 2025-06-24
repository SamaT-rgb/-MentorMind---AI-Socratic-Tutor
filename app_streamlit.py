# app_streamlit.py
import streamlit as st
import os
from tutor_logic import get_tutor_instance, get_available_personalities, LLM_MODEL_NAME

# --- Configuration for Image Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="MentorMind - AI Socratic Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {
        color: #1E88E5; /* Professional Blue */
        text-align: center;
        padding-bottom: 10px;
        font-family: 'Arial', sans-serif; /* Clean font */
    }
    .sidebar-title {
        color: #1E88E5;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar .css-1d391kg { 
        background-color: #f4f6f8; /* Lighter grey for sidebar */
    }
    .tutor-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .tutor-card img {
        border-radius: 50%;
        width: 100px; 
        height: 100px;
        object-fit: cover;
        margin-bottom: 12px;
        border: 3px solid #e0e0e0;
    }
    .tutor-name {
        font-size: 1.15em; /* Slightly smaller */
        font-weight: 600; /* Semibold */
        color: #333745; /* Darker text */
        margin-bottom: 5px;
    }
    .tutor-bio {
        font-size: 0.85em;
        color: #505050; /* Medium grey */
        line-height: 1.4;
    }
    /* More subtle chat message styling */
    .stChatMessage[data-testid="chatAvatarIcon-user"] + div div[data-testid="stMarkdownContainer"] {
        background-color: #e3f2fd; /* Lighter blue for user */
        border-radius: 18px;
        border-top-left-radius: 5px;
        padding: 10px 15px;
        margin-right: 50px; /* Give some space on the right */
    }
    .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div div[data-testid="stMarkdownContainer"] {
        background-color: #e8f5e9; /* Lighter green for assistant */
        border-radius: 18px;
        border-top-right-radius: 5px;
        padding: 10px 15px;
        margin-left: 50px; /* Give some space on the left */
    }
    /* Center the spinner text if possible */
    .stSpinner > div > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_resource(show_spinner=False) # Spinner handled manually in UI
def get_cached_tutor(personality_name_for_cache: str):
    tutor = get_tutor_instance(personality_name=personality_name_for_cache)
    if not tutor or not tutor.chain:
        return None
    return tutor

# --- Main App Structure ---

# Load personalities
available_personalities_dict = get_available_personalities()
all_personality_names = sorted(list(available_personalities_dict.keys()))

# Initialize session state variables
if "selected_personality" not in st.session_state:
    st.session_state.selected_personality = all_personality_names[0]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tutor_just_changed" not in st.session_state: # To manage initial message from new tutor
    st.session_state.tutor_just_changed = True


# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🌟 MentorMind Options</h2>", unsafe_allow_html=True)
    st.markdown("Your personal AI Socratic learning companion.")
    st.markdown("---")
    st.markdown("### Choose Your Tutor:")

    current_selection_index = all_personality_names.index(st.session_state.selected_personality)
    newly_selected_personality_name_from_ui = st.selectbox(
        "Select a Tutor Personality:",
        options=all_personality_names,
        index=current_selection_index,
        key="personality_selectbox_widget",
        help="Select a personality to guide your learning."
    )

    active_tutor = get_cached_tutor(newly_selected_personality_name_from_ui)

    st.markdown("---")
    if active_tutor:
        current_personality_key = active_tutor.get_current_personality_name() # Get actual name from tutor
        personality_details = available_personalities_dict.get(current_personality_key, {})
        
        st.markdown("<div class='tutor-card'>", unsafe_allow_html=True)
        image_filename = personality_details.get("image_filename")
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path):
                st.image(image_path) # CSS will handle width/height/border-radius
            else:
                st.caption(f"Img: {image_filename} missing")
        
        st.markdown(f"<div class='tutor-name'>{current_personality_key}</div>", unsafe_allow_html=True)
        
        bio = personality_details.get("bio", "No biography available for this tutor.")
        st.markdown(f"<div class='tutor-bio'>{bio}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Tutor could not be initialized. Check API key & model name.")
    
    st.markdown("---")
    if st.button("Clear Conversation History 🧹", key="clear_chat_button", use_container_width=True):
        st.session_state.messages = []
        get_cached_tutor.clear()
        st.session_state.tutor_just_changed = True # Trigger re-initialization message
        st.rerun()


# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>🎓 MentorMind - AI Socratic Tutor</h1>", unsafe_allow_html=True)

model_name_display = LLM_MODEL_NAME
if active_tutor:
    model_name_display = active_tutor.get_llm_model_name()
st.markdown(f"<p style='text-align: center; color: grey; margin-bottom: 20px;'>Interacting as: <b>{st.session_state.selected_personality}</b> (Model: {model_name_display})</p>", unsafe_allow_html=True)
# No extra <hr> here, CSS padding on title is enough

# Handle personality change and initialize messages
if st.session_state.selected_personality != newly_selected_personality_name_from_ui:
    st.session_state.selected_personality = newly_selected_personality_name_from_ui
    st.session_state.messages = [] # Clear previous messages
    st.session_state.tutor_just_changed = True # Signal that the tutor changed
    # The cache will be busted by get_cached_tutor(new_name)
    st.rerun()

# Display initial message from new tutor or if messages are empty
if active_tutor and (st.session_state.tutor_just_changed or not st.session_state.messages):
    initial_message = f"Hello! I am {active_tutor.get_current_personality_name()}. How can I guide your learning today?"
    # Add to history only if it's not already the first message (to avoid duplicates on fast reruns)
    if not st.session_state.messages or st.session_state.messages[0]["content"] != initial_message:
        st.session_state.messages.insert(0, {"role": "assistant", "content": initial_message})
    st.session_state.tutor_just_changed = False # Reset flag


# Display chat messages
if not active_tutor:
    st.error("Tutor is not available. Please check API key setup and selected model.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_prompt := st.chat_input(f"Ask {st.session_state.selected_personality}..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"{st.session_state.selected_personality} is thinking..."):
                response = active_tutor.get_response(user_prompt)
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
