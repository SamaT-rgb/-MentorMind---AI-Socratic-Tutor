# app_streamlit.py
import streamlit as st
# Import WITHOUT TUTOR_LLM_MODEL
from tutor_logic import get_tutor_instance, get_available_personalities, LLM_MODEL_NAME

# --- Page Configuration ---
st.set_page_config(page_title="MentorMind - AI Socratic Tutor", page_icon="🎓", layout="wide")

st.title("🎓 MentorMind - AI Socratic Tutor")

# --- Personality Selection ---
available_personalities_dict = get_available_personalities()
personality_names = list(available_personalities_dict.keys())

if "selected_personality" not in st.session_state:
    st.session_state.selected_personality = personality_names[0]

st.sidebar.header("Choose Your Tutor:")
current_selection_index = personality_names.index(st.session_state.selected_personality)

newly_selected_personality_name = st.sidebar.selectbox(
    "Select a Tutor Personality:",
    options=personality_names,
    index=current_selection_index,
    key="personality_selectbox_widget"
)

# --- Caching the Tutor Instance based on personality ---
@st.cache_resource
def get_cached_tutor_streamlit(personality_name_for_cache: str):
    print(f"Streamlit: Initializing/getting cached tutor for: {personality_name_for_cache}")
    tutor = get_tutor_instance(personality_name=personality_name_for_cache)
    if not tutor or not tutor.chain:
        st.error(f"Failed to initialize Socratic Tutor for personality: {personality_name_for_cache}. Check server logs for API key or other errors.")
        return None
    return tutor

# --- UI Updates based on selected tutor ---
active_tutor = get_cached_tutor_streamlit(newly_selected_personality_name)

if active_tutor:
    # Get the model name directly from the active tutor instance
    model_name_display = active_tutor.get_llm_model_name()
    st.caption(f"Guiding you as: **{active_tutor.get_current_personality_name()}** (Model: {model_name_display})")
else:
    # If tutor fails to initialize, use the LLM_MODEL_NAME imported from tutor_logic for the caption
    # This LLM_MODEL_NAME is the one tutor_logic *attempts* to use.
    st.caption(f"Guiding you as: **{newly_selected_personality_name}** (Model: {LLM_MODEL_NAME}) - Tutor Not Initialized")


# --- Handle Personality Change & Chat History Reset ---
if st.session_state.selected_personality != newly_selected_personality_name:
    st.session_state.selected_personality = newly_selected_personality_name
    st.session_state.streamlit_messages = [
        {"role": "assistant", "content": f"Hello! I am now {st.session_state.selected_personality}. How can I guide your learning today?"}
    ]
    st.rerun()


# --- Chat Interface ---
if active_tutor is None:
    st.error("Tutor could not be initialized. Please check the console/server logs for error messages (e.g., API key setup in .env).")
else:
    if "streamlit_messages" not in st.session_state:
        st.session_state.streamlit_messages = [
            {"role": "assistant", "content": f"Hello! I'm {active_tutor.get_current_personality_name()}. What would you like to learn or solve today?"}
        ]

    for message in st.session_state.streamlit_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input(f"Ask {st.session_state.selected_personality}..."):
        st.session_state.streamlit_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"{st.session_state.selected_personality} is thinking..."):
                response = active_tutor.get_response(user_prompt)
                st.markdown(response)
                st.session_state.streamlit_messages.append({"role": "assistant", "content": response})
