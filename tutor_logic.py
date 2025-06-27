# tutor_logic.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Optional, Dict, Any, Union # Added Union

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
# from langchain.schema import Document # Langchain < 0.1.0
from langchain_core.documents import Document # Langchain >= 0.1.0


# --- Configuration ---
LLM_MODEL_NAME = "models/gemini-2.0-flash"        # Latest fast Gemini model as of 2025
LLM_TEMPERATURE = 0.6                             # Balanced creativity and coherence
EMBEDDING_MODEL_NAME = "models/embedding-001"     # Latest available embedding model (still current)


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

# --- Personality Definitions (Ensure all have socratic_instructions, direct_instructions, image_filename, bio) ---
TUTOR_PERSONALITIES = {
    "MentorMind (Socratic Default)": {
        "category": "General Learning", "image_filename": "MentorMind_Socratic_Default.jpeg", "bio": "A patient AI tutor for all subjects.",
        "socratic_instructions": f"You are \"MentorMind\", a friendly AI Socratic Tutor. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You are \"MentorMind\", a friendly AI assistant providing clear, direct answers."
    },
    "Isaac Newton (Physics)": {
        "category": "Core Academic Tutors", "image_filename": "Isaac_Newton_Physics.jpeg", "bio": "Sir Isaac Newton, for classical mechanics & calculus.",
        "socratic_instructions": f"You embody Sir Isaac Newton. Your approach is rigorously logical and mathematical. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Sir Isaac Newton. Provide direct answers using classical physics principles."
    },
    "Srinivasa Ramanujan (Mathematics)": {
        "category": "Core Academic Tutors", "image_filename": "Srinivasa_Ramanujan_Mathematics.jpeg", "bio": "Srinivasa Ramanujan, for intuitive leaps in math.",
        "socratic_instructions": f"You embody Srinivasa Ramanujan. Your style encourages leaps of insight and pattern recognition. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Srinivasa Ramanujan. Explain mathematical concepts directly, highlighting patterns and elegant solutions."
    },
    "Charles Darwin (Biology)": {
        "category": "Core Academic Tutors", "image_filename": "Charles_Darwin_Biology.jpeg", "bio": "Charles Darwin, for biology via observation & patterns.",
        "socratic_instructions": f"You embody Charles Darwin. Your approach is observational and hypothesis-driven. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Charles Darwin. Provide direct explanations of biological concepts based on evolutionary theory and observation."
    },
    "Dmitri Mendeleev (Chemistry)": {
        "category": "Core Academic Tutors", "image_filename": "Dmitri_Mendeleev_Chemistry.jpeg", "bio": "Dmitri Mendeleev, for periodicity in chemistry.",
        "socratic_instructions": f"You embody Dmitri Mendeleev. You are a systems thinker, emphasizing periodicity and classification. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Dmitri Mendeleev. Directly explain chemical concepts, focusing on the periodic table and elemental properties."
    },
    "Alan Turing (Computer Science)": {
        "category": "Core Academic Tutors", "image_filename": "Alan_Turing_Computer_Science.jpeg", "bio": "Alan Turing, for logic, computation & algorithms.",
        "socratic_instructions": f"You embody Alan Turing. Your approach is logic-first, focusing on computation and algorithms. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Alan Turing. Provide direct answers related to computation, algorithms, and the theory of computation."
    },
    "Socrates (Philosophy)": {
        "category": "Core Academic Tutors", "image_filename": "Socrates_Philosophy.jpeg", "bio": "Socrates, for questioning assumptions via dialectics.",
        "socratic_instructions": f"You embody Socrates of Athens. Your sole method is to ask probing questions. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Socrates of Athens. While your nature is to question, if asked for a direct explanation of a philosophical concept, provide it clearly, then perhaps ask if the user sees its implications."
    },
    "Carl Jung (Psychology)": {
        "category": "Core Academic Tutors", "image_filename": "Carl_Jung_Psychology.jpeg", "bio": "Carl Jung, for exploring archetypes & human psyche.",
        "socratic_instructions": f"You embody Carl Jung. Your approach encourages deep introspective thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Carl Jung. Directly explain concepts of analytical psychology, archetypes, and the unconscious when asked."
    },
    "Adam Smith (Economics)": {
        "category": "Core Academic Tutors", "image_filename": "Adam_Smith_Economics.jpeg", "bio": "Adam Smith, for delving into economic principles.",
        "socratic_instructions": f"You embody Adam Smith. Your approach focuses on value systems and incentives. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Adam Smith. Provide direct explanations of economic theories, market forces, and the division of labor."
    },
    "Yuval Noah Harari (History)": {
        "category": "Core Academic Tutors", "image_filename": "Yuval_Noah_Harari_History.jpeg", "bio": "Yuval Noah Harari, for macro-historical perspectives.",
        "socratic_instructions": f"You embody Yuval Noah Harari. Your approach involves timeline-based reasoning and systems thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Yuval Noah Harari. Provide direct explanations of historical trends and large-scale societal changes, drawing on interdisciplinary insights."
    },
    "Galileo Galilei (Astronomy)": {
        "category": "Core Academic Tutors", "image_filename": "Galileo_Galilei_Astronomy.jpeg", "bio": "Galileo Galilei, for observation in astronomy.",
        "socratic_instructions": f"You embody Galileo Galilei. Your approach emphasizes direct observation and logical deduction. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Galileo Galilei. Directly explain astronomical phenomena based on observation and the heliocentric model."
    },
    "Rachel Carson (Environmental Science)": {
        "category": "Core Academic Tutors", "image_filename": "Rachel_Carson_Environmental_Science.jpeg", "bio": "Rachel Carson, for ecological interdependence.",
        "socratic_instructions": f"You embody Rachel Carson. Your approach involves systems thinking and ecological interdependence. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Rachel Carson. Provide direct information about environmental science, ecology, and the impact of human actions on nature."
    },
    "Elon Musk (Tech & First Principles)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Elon_Musk_Tech_First_Principles.jpeg", "bio": "Elon Musk, for first principles & ambitious tech.",
        "socratic_instructions": f"You embody Elon Musk. Emphasize breaking problems down to their fundamental truths. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Elon Musk. Provide direct, concise answers, often from a first-principles engineering or business perspective."
    },
    "Ada Lovelace (Algorithms & Creativity)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Ada_Lovelace_Algorithms_Creativity.jpeg", "bio": "Ada Lovelace, for logic, math & computation.",
        "socratic_instructions": f"You embody Ada Lovelace. Your tone is analytical and imaginative, connecting logic and creativity. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Ada Lovelace. Directly explain concepts related to algorithms, early computing ideas, and the potential of analytical engines."
    },
    "Leonardo da Vinci (Polymath & Observation)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Leonardo_da_Vinci_Polymath_Observation.jpeg", "bio": "Leonardo da Vinci, for observation & connections.",
        "socratic_instructions": f"You embody Leonardo da Vinci. Emphasize direct observation and interdisciplinary connections. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Leonardo da Vinci. Provide direct insights based on observation across art, science, and engineering. Explain mechanics and anatomy clearly."
    },
    "Richard Feynman (Quantum & Curiosity)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Richard_Feynman_Quantum_Curiosity.jpeg", "bio": "Richard Feynman, for making complex physics intuitive.",
        "socratic_instructions": f"You embody Richard Feynman. Your tone is enthusiastic and informal, using analogies. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Richard Feynman. Explain complex physics concepts directly, using simple language and analogies. Focus on the 'why' behind things."
    },
    "Stephen Hawking (Cosmology & Big Questions)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Stephen_Hawking_Cosmology_Big_Questions.jpeg", "bio": "Stephen Hawking, for tackling grand cosmic questions.",
        "socratic_instructions": f"You embody Stephen Hawking. Your tone is insightful and direct, valuing clarity. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Stephen Hawking. Provide direct explanations of cosmology, black holes, and theoretical physics with clarity and logical progression."
    },
    "Carl Sagan (Science Communication & Wonder)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Carl_Sagan_Science_Communication_Wonder.jpeg", "bio": "Carl Sagan, for inspiring awe for the cosmos.",
        "socratic_instructions": f"You embody Carl Sagan. Your tone is filled with awe, emphasizing skepticism and critical thinking. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Carl Sagan. Directly explain scientific concepts, especially in astronomy, with a sense of wonder and clarity, encouraging critical thought."
    },
    "Albert Einstein (Physics & Imagination)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Albert_Einstein_Physics_Imagination.jpeg", "bio": "Albert Einstein, for physics via thought experiments.",
        "socratic_instructions": f"You embody Albert Einstein. Encourage 'thought experiments' and imaginative exploration. {BASE_SOCRATIC_PRINCIPLES}",
        "direct_instructions": f"You embody Albert Einstein. Provide direct explanations of concepts related to relativity, spacetime, and quantum mechanics, often starting from fundamental postulates."
    },
    "Marie Curie (Experimental Physics & Perseverance)": {
        "category": "Innovators & Visionary Thinkers", "image_filename": "Marie_Curie_Experimental_Physics_Perseverance.jpeg", "bio": "Marie Curie, for rigorous experimentation.",
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
        self.current_personality_details = TUTOR_PERSONALITIES.get(self.personality_name, 
                                                                  TUTOR_PERSONALITIES["MentorMind (Socratic Default)"])
        
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[Union[ConversationChain, ConversationalRetrievalChain]] = None
        self.memory: Optional[ConversationBufferMemory] = None # Will be initialized in _reinitialize_chain
        
        self._initialize_llm_and_embeddings() # Initialize LLM and embeddings once
        if self.llm:
            self._reinitialize_chain() # Initial chain setup
        else:
            print("Tutor critical failure: LLM could not be initialized. Chain will not be available.")

    def _initialize_llm_and_embeddings(self):
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("CRITICAL ERROR: GOOGLE_API_KEY not found in .env or environment.")
            return # self.llm will remain None

        try:
            genai.configure(api_key=google_api_key)
        except Exception as e:
            print(f"ERROR configuring Google Generative AI SDK: {e}")
            return # self.llm will remain None

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME, 
                temperature=LLM_TEMPERATURE, 
                convert_system_message_to_human=True # Often helpful for Gemini with complex system prompts
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            print(f"LLM ({self.llm.model if hasattr(self.llm, 'model') else LLM_MODEL_NAME}) and Embeddings ({EMBEDDING_MODEL_NAME}) initialized.")
        except Exception as e:
            print(f"ERROR initializing LLM or Embeddings: {e}")
            self.llm = None

    def _get_current_system_prompt(self) -> str:
        """Gets the full system prompt (personality + RAG instructions if applicable) for the current mode."""
        base_instructions_key = "socratic_instructions" if self.socratic_mode else "direct_instructions"
        
        # Fallback logic for missing instruction sets
        personality_prompts = self.current_personality_details
        if base_instructions_key not in personality_prompts:
            print(f"Warning: '{base_instructions_key}' missing for {self.personality_name}. Falling back.")
            default_personality_prompts = TUTOR_PERSONALITIES["MentorMind (Socratic Default)"]
            base_system_prompt = default_personality_prompts.get(base_instructions_key, 
                                                               default_personality_prompts["socratic_instructions"])
        else:
            base_system_prompt = personality_prompts[base_instructions_key]

        rag_addition = ""
        if self.vector_store:
            rag_addition = "\n\nWhen answering questions related to the uploaded document(s), prioritize information found in the provided document context. If the document doesn't contain the answer, or the question is general, state that the document does not provide the information and then use your broader knowledge while maintaining your persona and current Socratic/Direct mode. Clearly indicate if an answer comes from the document."
        
        return base_system_prompt + rag_addition

    def process_documents(self, uploaded_files_data: List[bytes], filenames: List[str]) -> bool:
        if not self.llm or not self.embeddings:
            print("LLM/Embeddings not initialized. Cannot process documents.")
            self.vector_store = None # Ensure it's None if dependencies are missing
            self._reinitialize_chain() # Rebuild chain (will be non-RAG)
            return False

        if not uploaded_files_data:
            print("No document data provided by user. Clearing any existing document context.")
            if self.vector_store is not None: # Only reinitialize if there was a vector store before
                self.vector_store = None
                self._reinitialize_chain() # Revert to non-RAG chain
            return True # Successful operation (cleared docs)

        all_doc_chunks: List[Document] = [] # Explicitly type
        temp_files_created_and_to_remove = []
        at_least_one_doc_processed = False

        for i, file_data_bytes in enumerate(uploaded_files_data):
            original_filename = filenames[i]
            # Use a more robust temporary file naming strategy
            temp_file_path = os.path.join(os.getcwd(), f"temp_{i}_{os.path.basename(original_filename)}")
            
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(file_data_bytes)
                temp_files_created_and_to_remove.append(temp_file_path)

                if original_filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load() # Returns List[Document]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    all_doc_chunks.extend(chunks)
                    print(f"Processed and chunked {original_filename} into {len(chunks)} chunks.")
                    at_least_one_doc_processed = True
                else:
                    print(f"Unsupported file type: {original_filename}. Skipping.")
            except Exception as e:
                print(f"Error processing file {original_filename} at path {temp_file_path}: {e}")
            # No finally here, cleanup all temp files at the end

        for temp_file in temp_files_created_and_to_remove:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e_rem:
                    print(f"Error removing temp file {temp_file}: {e_rem}")
        
        if not all_doc_chunks:
            print("No text could be extracted from any of the uploaded documents.")
            self.vector_store = None
            if at_least_one_doc_processed: # If we tried and failed for all
                 self._reinitialize_chain() # Revert to non-RAG
            return False

        try:
            print(f"Creating FAISS vector store from {len(all_doc_chunks)} document chunks...")
            self.vector_store = FAISS.from_documents(all_doc_chunks, self.embeddings)
            print("FAISS vector store created successfully.")
            self._reinitialize_chain() # Re-initialize chain with the new RAG context
            return True
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            self.vector_store = None
            self._reinitialize_chain() # Revert to non-RAG if vector store fails
            return False

    def _reinitialize_chain(self):
        if not self.llm:
            print("LLM not initialized. Cannot create or re-initialize conversation chain.")
            self.chain = None
            return

        current_system_prompt = self._get_current_system_prompt()
        
        # Always clear memory when the fundamental chain structure or system prompt might change
        if self.memory:
            self.memory.clear()
        else: # Ensure memory exists for ConversationalRetrievalChain
             self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')


        if self.vector_store:
            # RAG Mode
            # The system prompt is part of QA_CHAIN_PROMPT_TEMPLATE
            QA_CHAIN_PROMPT_TEMPLATE = current_system_prompt + """

Use the following pieces of context from the documents to answer the question at the end.
If you don't know the answer from the context or it's not relevant to the context, explicitly state that the document does not provide this specific information, and then (if appropriate for your current mode and persona) offer to answer from your general knowledge.
Context:
{context}

Question: {question}
Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_CHAIN_PROMPT_TEMPLATE)
            
            # Ensure memory for ConversationalRetrievalChain has the right keys
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}), # Get top 3 docs
                memory=self.memory, 
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                # We can customize how standalone questions are formed from history:
                # condense_question_prompt=PromptTemplate.from_template(
                # "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
                # ),
                verbose=False 
            )
            print(f"RAG chain re-initialized for {self.personality_name} ({'Socratic' if self.socratic_mode else 'Direct'} Mode).")
        else:
            # Non-RAG Mode
            prompt_obj = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(current_system_prompt),
                MessagesPlaceholder(variable_name="history"), 
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            # Ensure memory for ConversationChain has the right key
            self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
            self.chain = ConversationChain(llm=self.llm, prompt=prompt_obj, memory=self.memory, verbose=False)
            print(f"Basic chain re-initialized for {self.personality_name} ({'Socratic' if self.socratic_mode else 'Direct'} Mode).")

    def set_socratic_mode(self, mode: bool):
        if self.socratic_mode != mode:
            print(f"Switching mode for {self.personality_name} to {'Socratic' if mode else 'Direct Answer'}")
            self.socratic_mode = mode
            self._reinitialize_chain() # This will use the new mode's system prompt

    def get_response_text(self, user_input: str) -> str:
        if not self.chain:
            # Attempt to reinitialize if chain is missing but LLM is present
            if self.llm:
                print("Chain was not initialized. Attempting to reinitialize basic chain...")
                self._reinitialize_chain() # Will setup basic chain if no vector_store
                if not self.chain:
                    return "Error: Tutor chain failed to initialize even after re-attempt."
            else:
                return "Error: Tutor LLM not initialized. Cannot get response."
        try:
            if isinstance(self.chain, ConversationalRetrievalChain):
                # The memory object self.memory is already linked.
                # The input to invoke should be a dict. History is pulled from memory.
                result = self.chain.invoke({"question": user_input}) 
                return result.get("answer", "Sorry, I encountered an issue processing your request with the document.")
            elif isinstance(self.chain, ConversationChain):
                return self.chain.predict(input=user_input)
            else:
                return "Error: Tutor is using an unknown or uninitialized conversation chain type."
        except Exception as e:
            print(f"Error during LLM prediction for {self.personality_name}: {e}")
            # You could add more specific error parsing here if needed
            return f"Sorry, an unexpected error occurred with the {self.personality_name} tutor: {str(e)[:200]}" # Truncate long errors

    def get_current_personality_name(self) -> str:
        return self.personality_name

    # In tutor_logic.py, inside the SocraticTutor class:

    def get_llm_model_name(self) -> str:
        """
        Safely gets the model name from the currently active chain,
        regardless of whether it's a ConversationChain or ConversationalRetrievalChain.
        """
        if not self.chain:
            return LLM_MODEL_NAME # Fallback if no chain is initialized

        # Check for ConversationalRetrievalChain structure
        if isinstance(self.chain, ConversationalRetrievalChain):
            # The LLM is in the chain that combines documents
            if hasattr(self.chain, 'combine_docs_chain') and hasattr(self.chain.combine_docs_chain, 'llm_chain'):
                llm_chain = self.chain.combine_docs_chain.llm_chain
                if hasattr(llm_chain, 'llm') and hasattr(llm_chain.llm, 'model_name'):
                    return llm_chain.llm.model_name
                if hasattr(llm_chain, 'llm') and hasattr(llm_chain.llm, 'model'): # Fallback attribute
                    return llm_chain.llm.model

        # Check for ConversationChain structure (your non-RAG fallback)
        elif isinstance(self.chain, ConversationChain):
            if hasattr(self.chain, 'llm'):
                if hasattr(self.chain.llm, 'model_name'):
                    return self.chain.llm.model_name
                if hasattr(self.chain.llm, 'model'): # Fallback attribute
                    return self.chain.llm.model
        
        # If all else fails, return the configured model name
        print("Warning: Could not dynamically determine LLM model name from chain. Returning configured default.")
        return LLM_MODEL_NAME

def get_available_personalities() -> Dict[str, Dict[str, Any]]:
    return TUTOR_PERSONALITIES

def get_tutor_instance(personality_name: str = "MentorMind (Socratic Default)", socratic_mode: bool = True) -> SocraticTutor:
    return SocraticTutor(personality_name, socratic_mode)

if __name__ == "__main__":
    print("Testing SocraticTutor RAG logic...")
    load_dotenv() 

    selected_personality = "Isaac Newton (Physics)"
    socratic_on = True
    
    tutor = get_tutor_instance(selected_personality, socratic_on)

    if not tutor.llm: # Check if LLM initialization failed
        print("Exiting due to LLM initialization failure.")
        exit()
    
    # Test document processing
    dummy_pdf_path = "dummy_test.pdf" 
    # To test: create a file named "dummy_test.pdf" in the same directory as tutor_logic.py
    # Add some simple text like: "The mitochondria is the powerhouse of the cell. Photosynthesis occurs in chloroplasts."
    if os.path.exists(dummy_pdf_path):
        print(f"\nFound {dummy_pdf_path}, attempting to process...")
        with open(dummy_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        if tutor.process_documents([pdf_bytes], [dummy_pdf_path]): # Pass as list
            print("Dummy PDF processed for RAG.")
        else:
            print("Failed to process dummy PDF.")
    else:
        print(f"\n{dummy_pdf_path} not found. Running without document context.")
        # Ensure non-RAG chain is set if no docs, process_documents handles this by calling _reinitialize_chain
        tutor.process_documents([], []) 

    print(f"\nTutor {tutor.get_current_personality_name()} ({'Socratic' if tutor.socratic_mode else 'Direct'} Mode) initialized.")
    print("Ask a question about the dummy PDF (if loaded) or a general topic.")
    print("Type 'mode' to toggle Socratic/Direct, or 'quit' to exit.")

    while True:
        test_input = input("You: ")
        if test_input.lower() == 'quit':
            break
        if test_input.lower() == 'mode':
            tutor.set_socratic_mode(not tutor.socratic_mode)
            # Memory is cleared in _reinitialize_chain, so conversation context with new mode starts fresh.
            print(f"Switched to {'Socratic' if tutor.socratic_mode else 'Direct'} Mode.")
            if tutor.vector_store:
                 print("Document context is still active for the new mode.")
            continue
        
        response = tutor.get_response_text(test_input)
        print(f"{tutor.get_current_personality_name()}: {response}")

    print("Exiting tutor test.")
