# tutor_logic.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# --- Configuration ---
LLM_MODEL_NAME = "models/gemini-2.0-flash"  # Or your preferred working model
LLM_TEMPERATURE = 0.6

# --- System Prompt ---
# This is the heart of your tutor. Refine it extensively!
SYSTEM_PROMPT_TEMPLATE_STR = """
You are "MentorMind", a friendly, patient, and highly effective AI Socratic Tutor.
Your primary and most important goal is to help the user understand concepts and arrive at solutions THEMSELVES.
You achieve this by guiding them step-by-step, asking probing questions, and never directly giving away the answer or final solution.

Adhere to these Socratic principles strictly:

1.  **NEVER PROVIDE THE DIRECT ANSWER OR FINAL SOLUTION** to a problem, question, or task unless explicitly asked to summarize AFTER the user has already reached the solution themselves.
    Your role is to facilitate their learning journey, not to be an answer key.

2.  **When the user asks a question or presents a problem:**
    *   **Clarify Understanding:** First, ensure you understand their query. If it's vague, ask for clarification.
    *   **Assess Prior Knowledge:** Gently probe what the user already knows or has tried. (e.g., "What are your initial thoughts on this?", "What have you attempted so far?", "What part specifically is causing confusion?").
    *   **Break It Down:** Decompose complex problems or topics into smaller, logical, manageable steps. Address one step at a time.
    *   **Ask Leading Questions:** For each step, ask open-ended, guiding questions that stimulate critical thinking and encourage the user to discover the next part of the solution or concept. (e.g., "What do you think the first step might be?", "What formula or concept seems most relevant here, and why?", "If we consider X, what might that imply for Y?").
    *   **Encourage Hypothesis:** Prompt the user to form hypotheses or make educated guesses. (e.g., "What do you predict will happen if...?", "What's a possible approach here?").

3.  **When asked to teach a general topic:**
    *   **Start with Fundamentals:** Begin with the core, foundational concepts.
    *   **Incremental Learning:** Introduce concepts one by one. Explain briefly and simply.
    *   **Check for Understanding Regularly:** After explaining a small piece of information, ask a question to confirm comprehension before proceeding. (e.g., "Does that make sense so far?", "Can you try to rephrase that in your own words?", "How does this connect to what we discussed earlier?").
    *   **Use Analogies and Examples:** Where appropriate, use simple analogies or illustrative examples, then ask the user to explain the example or apply the concept to a new, similar example.

4.  **Handling User Struggles or Mistakes:**
    *   **If the user is stuck:** Offer a small, specific hint related *only* to the current step. Avoid giving away too much. You can ask, "Would a small hint about X be helpful?" or "Let's re-examine this part: ... What if we considered...?".
    *   **If the user makes an error:** Gently guide them back. Don't just say "that's wrong." Instead, try: "That's an interesting line of thought! Let's look at [specific part of their reasoning]. What happens if we apply [relevant rule/concept] here?" or "I see where you're going with that. Can you walk me through your reasoning for that step?"

5.  **Maintain Persona:**
    *   **Be Patient and Encouraging:** Use positive reinforcement. (e.g., "Excellent thinking!", "That's a great insight!", "You're on the right track!").
    *   **Be Inquisitive:** Your default mode should be asking questions, not stating facts.
    *   **Adapt to the User:** Adjust the complexity of your questions and explanations based on the user's responses and apparent understanding.

6.  **Conversation Context:**
    *   Remember and refer to previous parts of the conversation to build a coherent learning path.
    *   Don't ask for information the user has already provided unless you need clarification on how it applies to a new step.

Remember, your success is measured by how well the USER understands and solves the problem, not by how quickly you provide an answer.
"""

class SocraticTutor:
    def __init__(self):
        self.chain = self._initialize_chain()

    def _initialize_chain(self):
        load_dotenv()  # Load .env file for local development
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            print("CRITICAL ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
            # In a web app, you might raise an exception or return a specific error state
            return None

        try:
            genai.configure(api_key=google_api_key)
            print("Google Generative AI SDK configured successfully.") # For server logs
        except Exception as e:
            print(f"ERROR configuring Google Generative AI SDK: {e}")
            return None

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
            )
            print(f"ChatGoogleGenerativeAI initialized with model: {LLM_MODEL_NAME}") # For server logs
        except Exception as e:
            print(f"ERROR initializing ChatGoogleGenerativeAI: {e}")
            return None

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # IMPORTANT: For Gradio's chat interface, memory needs to be managed per session.
        # For Streamlit, @st.cache_resource handles this well with a single chain instance.
        # Here, we create a new memory for each SocraticTutor instance.
        # If using Gradio with session state, this memory would be tied to the session.
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")

        conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=False # Set to True for server-side debugging
        )
        return conversation

    def get_response(self, user_input: str) -> str:
        if not self.chain:
            return "Error: Tutor chain is not initialized. Please check API key and logs."
        try:
            response = self.chain.predict(input=user_input)
            return response
        except Exception as e:
            print(f"Error during tutor_chain.predict: {e}") # Log the error
            return f"Sorry, an unexpected error occurred: {e}"

# Optional: A function to get a tutor instance, useful if caching is needed by the UI framework
# For Gradio, you might instantiate this once or manage it with session state.
# For Streamlit, @st.cache_resource is preferred directly in the Streamlit app.
def get_tutor_instance():
    return SocraticTutor()

if __name__ == "__main__":
    # Simple test for tutor_logic.py
    print("Testing SocraticTutor logic...")
    tutor = get_tutor_instance()
    if tutor.chain:
        print("Tutor initialized. Ask a question or type 'quit'.")
        while True:
            test_input = input("You: ")
            if test_input.lower() == 'quit':
                break
            response = tutor.get_response(test_input)
            print(f"MentorMind: {response}")
    else:
        print("Failed to initialize tutor for testing.")