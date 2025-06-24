# app_streamlit.py
import streamlit as st
import os # For constructing image paths
# Import LLM_MODEL_NAME for fallback caption if tutor fails to init
from tutor_logic import get_tutor_instance, get_available_personalities, LLM_MODEL_NAME

# --- Configuration for Image Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

# --- Page Configuration ---
st.set_page_config(page_title="MentorMind - AI Socratic Tutor", page_icon="🎓", layout="wide")

st.title("🎓 MentorMind - AI Socratic Tutor")

# --- Personality Selection ---
available_personalities_dict = get_available_personalities()
# Group personalities by category for the selectbox
grouped_options = {}
for name, details in available_personalities_dict.items():
    category = details.get("category", "Uncategorized")
    if category not in grouped_options:
        grouped_options[category] = []
    grouped_options[category].append(name)

# Flatten options for selectbox, but keep order for display if needed later or for optgroups
# For a simple selectbox, a flat list of names is easiest.
# For st.selectbox, optgroups are not directly supported.
# We can display categories as headers in the sidebar.
flat_personality_names = []
sorted_categories = sorted(grouped_options.keys())

st.sidebar.header("Choose Your Tutor:")
for category in sorted_categories:
    st.sidebar.markdown(f"**{category}**")
    for name in sorted(grouped_options[category]):
        # Use radio buttons for each category for better UX with many options
        if st.sidebar.radio(name, [name], key=f"radio_{name.replace(' ', '_')}", label_visibility="collapsed") == name:
            # If a radio button under a category is selected, update session state
            # This is a bit more complex than a single selectbox if we want to visually group AND select
            # For simplicity with many options, a single searchable selectbox is often better.
            # Let's revert to a single selectbox, and users can type to search.
            pass # Placeholder for more complex radio button logic

# Simpler single selectbox for all personalities
all_personality_names = sorted(list(available_personalities_dict.keys()))

if "selected_personality" not in st.session_state:
    st.session_state.selected_personality = all_personality_names[0] # Default to the first one

current_selection_index = all_personality_names.index(st.session_state.selected_personality)

newly_selected_personality_name = st.sidebar.selectbox(
    "Select a Tutor Personality:",
    options=all_personality_names, # Use the flat, sorted list
    index=current_selection_index,
    key="personality_selectbox_widget",
    help="Type to search for a specific tutor!"
)


# --- Caching the Tutor Instance based on personality ---
@st.cache_resource
def get_cached_tutor_streamlit(personality_name_for_cache: str):
    # print(f"Streamlit: Initializing/getting cached tutor for: {personality_name_for_cache}") # Less noisy
    tutor = get_tutor_instance(personality_name=personality_name_for_cache)
    if not tutor or not tutor.chain:
        # Error is printed in tutor_logic.py if API key is missing.
        # Streamlit error will be shown if tutor object itself is None after this.
        # st.error(f"Failed to initialize Socratic Tutor for {personality_name_for_cache}.") # Avoid double error
        return None
    return tutor

active_tutor = get_cached_tutor_streamlit(newly_selected_personality_name)

# --- Display Tutor Image in Sidebar ---
st.sidebar.markdown("---")
if active_tutor:
    st.sidebar.subheader(f"Current Tutor:")
    st.sidebar.markdown(f"**{active_tutor.get_current_personality_name()}**")
    personality_details = available_personalities_dict.get(active_tutor.get_current_personality_name())
    if personality_details and "image_filename" in personality_details:
        image_filename = personality_details["image_filename"]
        if image_filename:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            if os.path.exists(image_path):
                st.sidebar.image(image_path, width=150, use_column_width='auto') # use_column_width for sidebar
            else:
                st.sidebar.warning(f"Img not found: {image_filename}")
        # else: st.sidebar.caption("No image specified.") # Less verbose
    # else: st.sidebar.caption("Image details N/A.") # Less verbose
else:
    st.sidebar.subheader(f"Tutor: {newly_selected_personality_name}")
    st.sidebar.caption("(Initializing or error)")
st.sidebar.markdown("---")


# --- Main Area Caption ---
if active_tutor:
    model_name_display = active_tutor.get_llm_model_name()
    st.caption(f"Guiding you as: **{active_tutor.get_current_personality_name()}** (Model: {model_name_display})")
else:
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
    st.error("Tutor could not be initialized. Please check the console/server logs for error messages (e.g., API key setup in .env or model availability).")
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
