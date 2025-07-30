# MentorMind 🎓 — Your Personal AI Socratic Tutor

**MentorMind is an advanced, interactive AI learning platform designed to guide users through complex topics using the Socratic method.** Instead of providing instant answers, MentorMind helps you think critically and discover solutions for yourself. Choose from over 20 iconic personalities—from Albert Einstein to Leonardo da Vinci—each with a unique teaching style, and even upload your own study materials for a truly personalized learning session.

![MentorMind UI Screenshot]([https'://i.imgur.com/your-screenshot-url.png'](https://drive.google.com/file/d/1S10eVmBNzu9_IUdmpfbbOJprMEKBar4O/view?usp=sharing)) 

- **Project PDF:** [View a detailed presentation of the project's features and architecture.](https://raw.githubusercontent.com/SamaT-rgb/-MentorMind---AI-Socratic-Tutor/main/MentorMind%20-%20AI%20Socratic%20Tutor%20(Streamlit).pdf)

---


---

## 🚀 Key Features

*   **Multi-Persona Tutoring:** Learn from over 20 distinct AI personalities, including scientific pioneers, visionary innovators, and classical philosophers. Each tutor has a unique, persistent persona and teaching style.
*   **Dynamic Learning Modes:**
    *   **Socratic Guidance (Default):** The tutor asks leading questions to help you build intuition and arrive at your own conclusions.
    *   **Direct Answers:** Toggle this mode on to get clear, straightforward explanations from your chosen personality when you're stuck or need a quick fact-check.
*   **Analyze Your Own Documents (RAG):**
    *   Upload your own learning materials in various formats (**PDF, PPTX, DOCX, TXT**).
    *   MentorMind will read and understand the content, allowing you to ask questions, summarize key points, and learn directly from your own documents.
*   **Polished & Interactive UI:** Built with Streamlit for a clean, responsive, and user-friendly web experience.
*   **Robust Backend:** Powered by Google's Gemini models and architected with Langchain for seamless integration of conversational AI, memory, and document retrieval.

---

## 🛠️ Tech Stack & Architecture

| Layer                | Technology / Library                                                                            | Purpose                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Frontend**         | [Streamlit](https://streamlit.io/)                                                              | Building the interactive web interface and managing user session state.   |
| **Core Logic**       | Python                                                                                          | Orchestration, business logic, and server-side operations.                |
| **AI Orchestration** | [Langchain](https://www.langchain.com/)                                                         | Structuring LLM interactions, managing memory, and building RAG chains.   |
| **Language Model**   | [Google Gemini](https://ai.google.dev/) (e.g., `gemini-1.5-flash`)                                | The core AI engine for conversation, reasoning, and persona emulation.    |
| **Document Analysis (RAG)** | [Unstructured](https://unstructured.io/), [FAISS](https://faiss.ai/), [Google Embeddings](https://ai.google.dev/docs/embeddings_guide) | Loading & parsing documents, creating vector embeddings, and enabling semantic search. |

---

## ⚙️ Local Installation & Setup

Follow these steps to run MentorMind on your local machine.

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/SamaT-rgb/-MentorMind---AI-Socratic-Tutor.git
cd -MentorMind---AI-Socratic-Tutor
```

### 3. Create and Activate a Virtual Environment
This isolates the project's dependencies from your system.
```bash
# Create the environment
python -m venv myenv

# On Windows
myenv\Scripts\activate

# On macOS/Linux
source myenv/bin/activate
```

### 4. Install Dependencies
This command installs all necessary Python packages listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```
*(Note: This may take a few minutes as it includes libraries like PyTorch for FAISS and Unstructured).*

### 5. Add Your Google API Key
Create a `.env` file in the root directory of the project and add your API key:
```env
GOOGLE_API_KEY="your-google-api-key-here"
```
*(Optional for TTS: If you've set up Google Text-to-Speech, also add your `GOOGLE_APPLICATION_CREDENTIALS` path here.)*

### 6. Run the Streamlit App
Launch the application with the following command:
```bash
streamlit run app_streamlit.py
```
Your browser should automatically open to `http://localhost:8501`.

---

## 🔍 How It Works

1.  **Select a Tutor & Mode:** The user chooses a personality and a learning mode (Socratic or Direct) from the sidebar.
2.  **Initialize Tutor:** A `SocraticTutor` object is instantiated with a system prompt tailored to the selected persona and mode.
3.  **(Optional) Upload Documents:** The user can upload one or more documents.
    - **Backend Processing:** `tutor_logic.py` uses `UnstructuredFileLoader` to parse the files, chunks the text, and creates vector embeddings using Google's embedding model.
    - **Vector Store:** These embeddings are stored in an in-memory FAISS vector store.
    - **Chain Switch:** The conversation logic switches from a basic `ConversationChain` to a RAG-powered `ConversationalRetrievalChain`.
4.  **Engage in Conversation:**
    - The user's input is passed to the active Langchain chain.
    - If in RAG mode, the chain first retrieves relevant text chunks from the vector store and passes them as context to the LLM.
    - The Gemini model generates a response that adheres to the system prompt (persona, mode, and document context).
5.  **State Management:** Streamlit's session state and resource caching ensure that the conversation history, selected tutor, and document context are maintained throughout the user's session.

---

Happy Learning! 🌟
