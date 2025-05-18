# app_streamlit.py
import streamlit as st
from tutor_logic import SocraticTutor # Import the tutor class

# --- Page Configuration ---
st.set_page_config(page_title="MentorMind - AI Socratic Tutor (Streamlit)", page_icon="🎓")

# --- Caching the Tutor Instance ---
@st.cache_resource # Cache the SocraticTutor instance
def get_cached_tutor():
    """Gets a cached SocraticTutor instance."""
    tutor = SocraticTutor()
    if not tutor.chain: # Check if initialization failed within SocraticTutor
        st.error("Failed to initialize the Socratic Tutor. Please check logs/console for API key issues or other errors.")
        return None
    return tutor

# --- Streamlit App UI ---
st.title("🎓 MentorMind - AI Socratic Tutor")
st.caption(f"Powered by Langchain & Google Gemini ({SocraticTutor().chain.llm.model if SocraticTutor().chain and SocraticTutor().chain.llm else 'Model Unknown'})") # Display model if accessible

# Initialize or get the tutor session from cache
tutor = get_cached_tutor()

if tutor is None:
    st.warning("Tutor could not be initialized. Please check the console for error messages (e.g., API key).")
else:
    # Initialize chat history in session state if it doesn't exist
    if "streamlit_messages" not in st.session_state: # Use a unique key for streamlit messages
        st.session_state.streamlit_messages = [
            {"role": "assistant", "content": "Hello! I'm MentorMind. What would you like to learn or solve today?"}
        ]

    # Display chat messages from history
    for message in st.session_state.streamlit_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_prompt := st.chat_input("Ask MentorMind..."):
        # Add user message to chat history
        st.session_state.streamlit_messages.append({"role": "user", "content": user_prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("MentorMind is thinking..."):
                response = tutor.get_response(user_prompt) # Use the get_response method
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.streamlit_messages.append({"role": "assistant", "content": response})