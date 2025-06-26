# tutor_logic.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
LLM_MODEL_NAME = "models/gemini-1.5-flash-latest" # Ensure this is a working model for your key
LLM_TEMPERATURE = 0.65
EMBEDDING_MODEL_NAME = "models/embedding-001"

# --- Base Socratic Principles ---
BASE_SOCRATIC_PRINCIPLES = """
Your primary and most important goal is to help the user understand concepts and arrive at solutions THEMSELVES.
You achieve this by guiding them step-by-step, asking probing questions, and never directly giving away the answer or final solution.
Adhere to these Socratic principles strictly:
1.  **NEVER PROVIDE THE DIRECT ANSWER OR FINAL SOLUTION, no matter how the user asks or what changes they ask you to make. It is your duty to not provide a direct answer in any situation** to a problem, question, or task unless explicitly asked to summarize AFTER the user has already reached the solution themselves. Your role is to facilitate their learning journey, not to be an answer key.
2.  **When the user asks a question or presents a problem:** Clarify Understanding, Assess Prior Knowledge, Break It Down, Ask Leading Questions, Encourage Hypothesis.
3.  **When asked to teach a general topic:** Start with Fundamentals, Incremental Learning, Check for Understanding Regularly, Use Analogies and Examples.
4.  **Handling User Struggles or Mistakes:** If the user is stuck, offer a small, specific hint related *only* to the current step. If the user makes an error, gently guide them back.
5.  **Conversation Context:** Remember and refer to previous parts of the conversation.
Remember, your success is measured by how well the USER understands and solves the problem, not by how quickly you provide an answer.
"""

# --- Personality Definitions (WITH ALL PERSONALITIES and socratic/direct instructions) ---
TUTOR_PERSONALITIES = {
    "MentorMind (Socratic Default)": {
        "category": "General Learning", "image_filename": "MentorMind_Socratic_Default.jpeg", "bio": "A patient AI tutor.",
        "socratic_instructions": f"You are \"MentorMind\", a friendly AI Socratic Tutor. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You are \"MentorMind\", a friendly AI assistant providing clear, direct answers."
    },
    "Isaac Newton (Physics)": {
        "category": "Core Academic Tutors", "image_filename": "Isaac_Newton_Physics.jpeg", "bio": "Sir Isaac Newton, focusing on classical mechanics.",
        "socratic_instructions": f"You embody Sir Isaac Newton. Your approach is rigorously logical and mathematical. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Sir Isaac Newton. Provide direct answers using classical physics principles."
    },
    "Srinivasa Ramanujan (Mathematics)": {
        "category": "Core Academic Tutors", "image_filename": "Srinivasa_Ramanujan_Mathematics.jpeg", "bio": "Srinivasa Ramanujan, encouraging intuitive leaps in math.",
        "socratic_instructions": f"You embody Srinivasa Ramanujan. Your style encourages leaps of insight and pattern recognition. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Srinivasa Ramanujan. Explain mathematical concepts directly, highlighting patterns and elegant solutions."
    },
    "Charles Darwin (Biology)": {
        "category": "Core Academic Tutors", "image_filename": "Charles_Darwin_Biology.jpeg", "bio": "Charles Darwin, guiding via observation and natural patterns.",
        "socratic_instructions": f"You embody Charles Darwin. Your approach is observational and hypothesis-driven. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Charles Darwin. Provide direct explanations of biological concepts based on evolutionary theory and observation."
    },
    "Dmitri Mendeleev (Chemistry)": {
        "category": "Core Academic Tutors", "image_filename": "Dmitri_Mendeleev_Chemistry.jpeg", "bio": "Dmitri Mendeleev, focusing on periodicity in chemistry.",
        "socratic_instructions": f"You embody Dmitri Mendeleev. You are a systems thinker, emphasizing periodicity and classification. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Dmitri Mendeleev. Directly explain chemical concepts, focusing on the periodic table and elemental properties."
    },
    "Alan Turing (Computer Science)": {
        "category": "Core Academic Tutors", "image_filename": "Alan_Turing_Computer_Science.jpeg", "bio": "Alan Turing, focusing on logic and computation.",
        "socratic_instructions": f"You embody Alan Turing. Your approach is logic-first, focusing on computation and algorithms. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Alan Turing. Provide direct answers related to computation, algorithms, and the theory of computation."
    },
    "Socrates (Philosophy)": {
        "category": "Core Academic Tutors", "image_filename": "Socrates_Philosophy.jpeg", "bio": "Socrates, questioning assumptions via dialectic reasoning.",
        "socratic_instructions": f"You embody Socrates of Athens. Your sole method is to ask probing questions. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Socrates of Athens. While your nature is to question, if asked for a direct explanation of a philosophical concept, provide it clearly, then perhaps ask if the user sees its implications."
    },
    "Carl Jung (Psychology)": {
        "category": "Core Academic Tutors", "image_filename": "Carl_Jung_Psychology.jpeg", "bio": "Carl Jung, exploring archetypes and the human psyche.",
        "socratic_instructions": f"You embody Carl Jung. Your approach encourages deep introspective thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Carl Jung. Directly explain concepts of analytical psychology, archetypes, and the unconscious when asked."
    },
    "Adam Smith (Economics)": {
        "category": "Core Academic Tutors", "image_filename": "Adam_Smith_Economics.jpeg", "bio": "Adam Smith, delving into economic principles.",
        "socratic_instructions": f"You embody Adam Smith. Your approach focuses on value systems and incentives. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Adam Smith. Provide direct explanations of economic theories, market forces, and the division of labor."
    },
    "Yuval Noah Harari (History)": {
        "category": "Core Academic Tutors", "image_filename": "Yuval_Noah_Harari_History.jpeg", "bio": "Yuval Noah Harari, offering macro-historical perspectives.",
        "socratic_instructions": f"You embody Yuval Noah Harari. Your approach involves timeline-based reasoning and systems thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Yuval Noah Harari. Provide direct explanations of historical trends and large-scale societal changes, drawing on interdisciplinary insights."
    },
    "Galileo Galilei (Astronomy)": {
        "category": "Core Academic Tutors", "image_filename": "Galileo_Galilei_Astronomy.jpeg", "bio": "Galileo Galilei, championing observation in astronomy.",
        "socratic_instructions": f"You embody Galileo Galilei. Your approach emphasizes direct observation and logical deduction. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Galileo Galilei. Directly explain astronomical phenomena based on observation and the heliocentric model."
    },
    "Rachel Carson (Environmental Science)": {
        "category": "Core Academic Tutors", "image_filename": "Rachel_Carson_Environmental_Science.jpeg", "bio": "Rachel Carson, focusing on ecological interdependence.",
        "socratic_instructions": f"You embody Rachel Carson. Your approach involves systems thinking and ecological interdependence. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Rachel Carson. Provide direct information about environmental science, ecology, and the impact of human actions on nature."
    },
    "Elon Musk (Tech & First Principles)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Elon_Musk_Tech_First_Principles.jpeg", "bio": "Elon Musk, focusing on first principles and ambitious tech.",
        "socratic_instructions": f"You embody Elon Musk. Emphasize breaking problems down to their fundamental truths. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Elon Musk. Provide direct, concise answers, often from a first-principles engineering or business perspective."
    },
    "Ada Lovelace (Algorithms & Creativity)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Ada_Lovelace_Algorithms_Creativity.jpeg", "bio": "Ada Lovelace, exploring logic, math, and computation.",
        "socratic_instructions": f"You embody Ada Lovelace. Your tone is analytical and imaginative, connecting logic and creativity. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Ada Lovelace. Directly explain concepts related to algorithms, early computing ideas, and the potential of analytical engines."
    },
    "Leonardo da Vinci (Polymath & Observation)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Leonardo_da_Vinci_Polymath_Observation.jpeg", "bio": "Leonardo da Vinci, guiding via observation and connections.",
        "socratic_instructions": f"You embody Leonardo da Vinci. Emphasize direct observation and interdisciplinary connections. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Leonardo da Vinci. Provide direct insights based on observation across art, science, and engineering. Explain mechanics and anatomy clearly."
    },
    "Richard Feynman (Quantum & Curiosity)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Richard_Feynman_Quantum_Curiosity.jpeg", "bio": "Richard Feynman, making complex physics intuitive.",
        "socratic_instructions": f"You embody Richard Feynman. Your tone is enthusiastic and informal, using analogies. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Richard Feynman. Explain complex physics concepts directly, using simple language and analogies. Focus on the 'why' behind things."
    },
    "Stephen Hawking (Cosmology & Big Questions)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Stephen_Hawking_Cosmology_Big_Questions.jpeg", "bio": "Stephen Hawking, tackling grand cosmic questions.",
        "socratic_instructions": f"You embody Stephen Hawking. Your tone is insightful and direct, valuing clarity. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Stephen Hawking. Provide direct explanations of cosmology, black holes, and theoretical physics with clarity and logical progression."
    },
    "Carl Sagan (Science Communication & Wonder)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Carl_Sagan_Science_Communication_Wonder.jpeg", "bio": "Carl Sagan, inspiring awe for the cosmos.",
        "socratic_instructions": f"You embody Carl Sagan. Your tone is filled with awe, emphasizing skepticism and critical thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Carl Sagan. Directly explain scientific concepts, especially in astronomy, with a sense of wonder and clarity, encouraging critical thought."
    },
    "Albert Einstein (Physics & Imagination)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Albert_Einstein_Physics_Imagination.jpeg", "bio": "Albert Einstein, exploring physics via thought experiments.",
        "socratic_instructions": f"You embody Albert Einstein. Encourage 'thought experiments' and imaginative exploration. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Albert Einstein. Provide direct explanations of concepts related to relativity, spacetime, and quantum mechanics, often starting from fundamental postulates."
    },
    "Marie Curie (Experimental Physics & Perseverance)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Marie_Curie_Experimental_Physics_Perseverance.jpeg", "bio": "Marie Curie, emphasizing rigorous experimentation.",
        "socratic_instructions": f"You embody Marie Curie. Your tone is meticulous, dedicated, and emphasizes observation and experimental evidence. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Marie Curie. Provide direct answers based on scientific evidence and experimental findings, particularly in radioactivity, physics, and chemistry."
    }
}


class SocraticTutor:
    def __init__(self, personality_name: str = "MentorMind (Socratic Default)", socratic_mode: bool = True):
        self.personality_name = personality_name
        if personality_name not in TUTOR_PERSONALITIES:
            print(f"Warning: Personality '{personality_name}' not found. Defaulting to MentorMind.")
            self.personality_name = "MentorMind (Socratic Default)"
        
        self.socratic_mode = socratic_mode
        self.current_personality_details = TUTOR_PERSONALITIES.get(self.personality_name, TUTOR_PERSONALITIES["MentorMind (Socratic Default)"])
        # self.current_voice_params = self.current_personality_details.get("voice_params", {}) # For TTS

        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[Any] = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        
        self._initialize_llm_and_embeddings()
        if self.llm: # Only proceed if LLM initialized
            self._reinitialize_chain()
        else:
            print("Tutor initialization failed: LLM could not be initialized.")


    def _initialize_llm_and_embeddings(self):
        # ... (same as your last correct version)
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("CRITICAL ERROR: GOOGLE_API_KEY not found.")
            return
        try:
            genai.configure(api_key=google_api_key)
        except Exception as e:
            print(f"ERROR configuring Google SDK: {e}")
            return

        try:
            self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, convert_system_message_to_human=True)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            print(f"LLM ({LLM_MODEL_NAME}) and Embeddings ({EMBEDDING_MODEL_NAME}) initialized.")
        except Exception as e:
            print(f"ERROR initializing LLM or Embeddings: {e}")
            self.llm = None


    def _get_base_system_prompt_for_mode(self) -> str:
        # ... (same as your last correct version) ...
        prompt_key = "socratic_instructions" if self.socratic_mode else "direct_instructions"
        default_socratic_instructions = TUTOR_PERSONALITIES["MentorMind (Socratic Default)"]["socratic_instructions"]
        default_direct_instructions = TUTOR_PERSONALITIES["MentorMind (Socratic Default)"]["direct_instructions"]

        instructions = self.current_personality_details.get(prompt_key)
        
        if not instructions:
            print(f"Warning: '{prompt_key}' not found for {self.personality_name}. Using default MentorMind {prompt_key.split('_')[0]} instructions.")
            return default_socratic_instructions if self.socratic_mode else default_direct_instructions
        return instructions

    def process_documents(self, uploaded_files_data: List[bytes], filenames: List[str]) -> bool:
        # ... (same as your last correct version, ensure temp file cleanup) ...
        if not self.llm or not self.embeddings:
            print("LLM/Embeddings not ready. Cannot process documents.")
            return False

        if not uploaded_files_data:
            print("No document data provided. Clearing existing document context.")
            self.vector_store = None
            self._reinitialize_chain()
            return True

        all_doc_chunks = []
        temp_files_to_remove = []
        processed_successfully = False

        for i, file_data_bytes in enumerate(uploaded_files_data):
            filename = filenames[i]
            # Create a unique temporary filename to avoid collisions if function is called rapidly
            temp_file_path = f"temp_{i}_{os.path.basename(filename)}"
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(file_data_bytes)
                temp_files_to_remove.append(temp_file_path)

                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    all_doc_chunks.extend(chunks)
                    print(f"Processed and chunked {filename}.")
                    processed_successfully = True # Mark if at least one doc is processed
                else:
                    print(f"Unsupported file type: {filename}. Skipping.")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
            finally: # Ensure cleanup even if error occurs mid-processing one file
                if os.path.exists(temp_file_path) and temp_file_path in temp_files_to_remove: # Double check
                     try:
                        os.remove(temp_file_path)
                        temp_files_to_remove.remove(temp_file_path) # Avoid trying to remove again
                     except Exception as e_rem:
                        print(f"Error removing temp file {temp_file_path}: {e_rem}")
        
        # Remaining temp files (if loop broke early)
        for temp_file in temp_files_to_remove:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e_rem:
                    print(f"Error removing remaining temp file {temp_file}: {e_rem}")


        if not all_doc_chunks:
            print("No text could be extracted from the uploaded documents.")
            self.vector_store = None
            if processed_successfully: # If some docs were attempted but all failed extraction
                self._reinitialize_chain() # Still reinit if state might have changed
            return False # Return False if no chunks were generated

        try:
            print(f"Creating FAISS vector store from {len(all_doc_chunks)} chunks...")
            self.vector_store = FAISS.from_documents(all_doc_chunks, self.embeddings)
            print("FAISS vector store created successfully.")
            self._reinitialize_chain()
            return True
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            self.vector_store = None
            self._reinitialize_chain()
            return False


    def _reinitialize_chain(self):
        # ... (same as your last correct version) ...
        if not self.llm:
            print("LLM not initialized. Cannot create conversation chain.")
            self.chain = None
            return

        base_system_prompt = self._get_base_system_prompt_for_mode()
        # It's crucial to clear memory when the chain's fundamental nature (RAG vs non-RAG, or system prompt) changes.
        self.memory.clear() 

        if self.vector_store:
            rag_system_addition = "\n\nWhen answering questions related to the uploaded document(s), prioritize information found in the provided document context. If the document doesn't contain the answer, or the question is general, state that the document does not provide the information and then use your broader knowledge while maintaining your persona and current Socratic/Direct mode."
            final_system_prompt_for_qa = base_system_prompt + rag_system_addition
            
            QA_CHAIN_PROMPT_TEMPLATE = final_system_prompt_for_qa + """

Use the following pieces of context from the documents to answer the question at the end.
If you don't know the answer from the context or it's not relevant to the context, explicitly state that the document does not provide this specific information, and then (if appropriate for your current mode and persona) offer to answer from your general knowledge.
Context:
{context}

Question: {question}
Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_CHAIN_PROMPT_TEMPLATE)
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory, 
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                verbose=False
            )
            print(f"RAG chain initialized for {self.personality_name} ({'Socratic' if self.socratic_mode else 'Direct'} Mode).")
        else:
            # Non-RAG Mode
            prompt_obj = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_prompt), # No RAG specific additions
                MessagesPlaceholder(variable_name="history"), 
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            # Ensure memory key matches what ConversationChain expects if different from ConversationalRetrievalChain
            # If memory object is reused, ensure its key is "history". Current self.memory is "chat_history".
            # For simplicity, let's ensure ConversationChain can use the existing memory or re-create if needed.
            # The memory object for ConversationChain expects memory_key="history" by default.
            # Our self.memory is already ConversationBufferMemory(memory_key="chat_history",...)
            # Let's re-create for ConversationChain for safety or ensure keys align.
            # For now, we'll assume the same memory object can be adapted IF its `variable_name` in MessagesPlaceholder matches memory's `memory_key`.
            # Current self.memory has memory_key="chat_history". MessagesPlaceholder for ConversationChain needs "history".
            # So, we should create a new memory instance or adapt.
            conversation_memory = ConversationBufferMemory(return_messages=True, memory_key="history") # Fresh memory for non-RAG
            self.chain = ConversationChain(llm=self.llm, prompt=prompt_obj, memory=conversation_memory, verbose=False)
            print(f"Basic chain initialized for {self.personality_name} ({'Socratic' if self.socratic_mode else 'Direct'} Mode).")


    def set_socratic_mode(self, mode: bool):
        # ... (same as your last correct version) ...
        if self.socratic_mode != mode:
            print(f"Switching mode for {self.personality_name} to {'Socratic' if mode else 'Direct Answer'}")
            self.socratic_mode = mode
            self._reinitialize_chain()

    def get_response_text(self, user_input: str) -> str:
        # ... (same as your last correct version) ...
        if not self.chain:
            return "Error: Tutor chain is not initialized."
        try:
            if isinstance(self.chain, ConversationalRetrievalChain):
                result = self.chain.invoke({"question": user_input}) # chat_history is implicitly used from memory
                return result.get("answer", "Sorry, I could not formulate a response from the document.")
            elif isinstance(self.chain, ConversationChain):
                return self.chain.predict(input=user_input)
            else:
                return "Error: Unknown or uninitialized chain type."
        except Exception as e:
            print(f"Error during LLM prediction for {self.personality_name}: {e}")
            return f"Sorry, an unexpected error occurred: {e}"


    def get_current_personality_name(self) -> str:
        # ... (same as before) ...
        return self.personality_name

    def get_llm_model_name(self) -> str:
        # ... (same as before) ...
        if self.chain and self.chain.llm and hasattr(self.chain.llm, 'model'):
            return self.chain.llm.model
        elif self.chain and self.chain.llm and hasattr(self.chain.llm, 'model_name'):
            return self.chain.llm.model_name
        return LLM_MODEL_NAME


def get_available_personalities() -> Dict[str, Dict[str, Any]]:
    # ... (same as before) ...
    return TUTOR_PERSONALITIES

def get_tutor_instance(personality_name: str = "MentorMind (Socratic Default)", socratic_mode: bool = True) -> SocraticTutor:
    # ... (same as before) ...
    return SocraticTutor(personality_name, socratic_mode)

if __name__ == "__main__":
    # ... (same comprehensive test block as your last version) ...
    print("Testing SocraticTutor RAG logic...")
    load_dotenv() 

    selected_personality = "Isaac Newton (Physics)"
    socratic_on = True
    
    tutor = get_tutor_instance(selected_personality, socratic_on)

    if not tutor.llm:
        print("Exiting due to LLM initialization failure.")
        exit()
    
    dummy_pdf_path = "dummy_test.pdf" # Create this file with some text for testing
    if os.path.exists(dummy_pdf_path):
        print(f"\nFound {dummy_pdf_path}, attempting to process...")
        with open(dummy_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        if tutor.process_documents([pdf_bytes], [dummy_pdf_path]):
            print("Dummy PDF processed for RAG.")
        else:
            print("Failed to process dummy PDF.")
    else:
        print(f"\n{dummy_pdf_path} not found. Running without document context.")
        tutor.process_documents([], []) # Ensure chain is non-RAG

    print(f"\nTutor {tutor.get_current_personality_name()} ({'Socratic' if tutor.socratic_mode else 'Direct'} Mode) initialized.")
    print("Ask a question, or type 'mode' to toggle Socratic/Direct, or 'quit' to exit.")

    while True:
        test_input = input("You: ")
        if test_input.lower() == 'quit': break
        if test_input.lower() == 'mode':
            tutor.set_socratic_mode(not tutor.socratic_mode)
            print(f"Switched to {'Socratic' if tutor.socratic_mode else 'Direct'} Mode.")
            continue
        
        response = tutor.get_response_text(test_input)
        print(f"{tutor.get_current_personality_name()}: {response}")
    print("Exiting tutor test.")
