
# MentorMind 🧠 — AI Socratic Tutor(still updating for now we have gemini 2.0 and have to improve UI aswell plus we have to add for image analysis)

MentorMind is an interactive AI-powered tutor designed to guide learners through reflective thinking using Socratic-style questioning. Powered by Streamlit and Gemini 2.0, it encourages users to understand concepts through meaningful conversation — not just passive answers.

---

## 🌐 Video Demo

📽️ [Watch Demo Video](https://drive.google.com/file/d/1NPsYcbIDUAdt_zL6PhbXIBJugoTuS6wE/view?usp=drive_link)

---

## 🚀 Features

-  AI-powered Socratic dialogue with Gemini 2.0  
-  Real-time, interactive chat sessions that challenge your thinking  
-  Encourages critical thinking instead of direct answers  
-  Flexible session types: conceptual walkthroughs, problem-solving  
-  Curated resource recommendations powered by LangChain  
-  Seamless and fast interface using Streamlit  

---

## 🛠 Tech Stack Overview

| Layer        | Technology Used                                                           |
|--------------|----------------------------------------------------------------------------|
| Interface    |  [Streamlit](https://streamlit.io/) – for building the web interface    |
| Core Logic   |  Python – scripting and orchestration                                    |
| AI Engine    |  Gemini 2.0 – for intelligent Socratic dialogues                         |
| Libraries    |  Pandas, Streamlit Chat UI, LangChain – for data processing and chat UX |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/SamaT-rgb/-MentorMind---AI-Socratic-Tutor.git
cd -MentorMind---AI-Socratic-Tutor
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Then open your browser and visit: [http://localhost:8501](http://localhost:8501)

---

## 🔍 How It Works

1. User inputs a question or topic.
2. MentorMind triggers a Gemini-powered prompt chain via LangChain.
3. The AI responds with thought-provoking questions, not direct answers.
4. If the user is stuck, MentorMind offers hints or curated learning resources.
5. This continues until the user gains clarity and deeper understanding.

This approach mirrors a thoughtful tutor — guiding users toward discovery rather than rote learning.

---

## 💬 Sample Interaction

**User:** "What is backpropagation in neural networks?"  
**MentorMind:** "What happens when a neural network makes an incorrect prediction? Can we trace the error back?"  
**User:** "Yes, through the layers."  
**MentorMind:** "Exactly! How do you think we adjust the weights in each layer using that error?"

---

Happy Learning! 🌟
