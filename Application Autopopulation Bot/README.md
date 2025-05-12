## 📚 Table of Contents
- [What It Does](#🚀-what-it-does)
- [Project Structure](#📁-project-structure)
- [Setup & Run Instructions](#🛠-setup--run-instructions)
- [Application Flow](#✍️-application-flow)
- [Features](#📋-features)
- [Notes](#⚠️-notes)
- [Technologies Used](#🧠-technologies-used)
- [Example Questions & Modification](#✅-example-questions--modification)
---


# 📝 Application Autopopulation Bot

This is a Streamlit-based tool that assists users in filling out **seed grant applications** by automatically generating draft answers from a company description. It leverages the **LLaMA 3.1** model (via [Ollama](https://ollama.com/)) and validates its output using **zero-shot NLI (Natural Language Inference)** to minimize hallucinations.


## 🚀 What It Does

1. **You provide a company description** and click the *Generate Answers* button.
2. The system uses an AI model to generate tailored responses to a predefined list of common application questions.
3. It validates each response against your input using a secondary NLI model to detect hallucinations.
4. It rewrites or rejects hallucinated content.
5. You can edit the answers manually, save the answers, and **export everything as a clean PDF**.


## 📁 Project Structure

```plaintext
📁 Application Autopopulation Bot/
├── auto_population.py         # Main Streamlit app with logic for QA generation, hallucination filtering, and PDF export
├── requirements.txt           # All required dependencies
├── charlotte_logo.png         # Logo shown in sidebar
└── README.md                  # This file
```


## 🛠 Setup & Run Instructions

Install [Python 3.12.2](https://www.python.org/downloads/) and follow these steps to run the application:

```bash
# 1. Navigate to the project directory
cd path/to/project-directory

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama (https://ollama.com/download)
# Follow instructions for your OS to install Ollama

# 5. Pull the LLaMA 3.1 model
ollama pull llama3.1

# 6. Run the Streamlit app
streamlit run auto_population.py
```


## ✍️ Application Flow

### 🧾 Step 1: Enter Company Description

Users input a detailed company description on the first screen. This serves as the **source of truth** for all generated answers.


### 🧠 Step 2: Generate Answers

When "Generate Answers" is clicked:
- A series of **predefined business questions** are asked.
- The **LLaMA 3.1 model** generates answers based solely on your company description.
- Each response is **split into text chunks** and run through a **zero-shot NLI model** (`facebook/bart-large-mnli`) to check if it aligns with the company description.
- Chunks with weak alignment are rewritten up to 2 times. If they still fail, the answer is replaced with `"Information not provided"`.


### ✏️ Step 3: View / Edit Answers

After generation:
- You can switch to the **"View/Edit Answers"** tab.
- Manually refine the answers in text boxes.

Click **Save Answers** to:
- Save your changes
- Export your answers to a clean, printable **PDF file** (`seed_grant_application.pdf`)

---

#  🧩 Additional Information

## 📋 Features

- GPU usage check with fallback
- Automated hallucination filtering and correction
- Human-readable prompts for model generation
- Editable fields and PDF export
- Streamlit-based UI with simple navigation


## ⚠️ Notes

- This app relies heavily on GPU performance. If no GPU is detected, it will exit.
- All core logic, including UI, generation, and PDF export, is combined into a single file (`auto_population.py`).
- Generated content may still require review by domain experts.


## 🧠 Technologies Used

- **Streamlit** – UI
- **LangChain + LLaMA 3.1 via Ollama** – Generation
- **Hugging Face Transformers (BART-MNLI)** – Fact validation
- **FPDF** – PDF export
- **NLTK** – Tokenization


## ✅ Example Questions & Modification

```python
questions = [
    "What is the elevator pitch?",
    "What problem is the company solving?",
    "What is the company's solution to the problem they are solving?",
    ...
]
```

You can modify these directly inside `auto_population.py` in the `questions = [...]` list.  

---

## 🧠 Agentic AI Behavior in This Application

This app exhibits **agentic AI characteristics** through its autonomous decision-making and flow control during answer generation:

- **Goal-Oriented Behavior**: Once the user provides a company description and clicks "Generate Answers," the system autonomously answers a predefined set of questions without further prompts.

- **Multi-Step Reasoning Loop**: For each generated answer, the AI:
  1. **Validates factual grounding** using a natural language inference (NLI) model.
  2. **Detects hallucinated text** by comparing each chunk of the answer to the original company description.
  3. **Decides independently** whether to regenerate the answer (up to 2 retries), using LLM-driven rewriting to remove or correct hallucinated content.

- **Self-Correcting Mechanism**: If the model fails to produce a grounded answer after maximum retries, it outputs a fallback: `"Information not found"`.

Together, these behaviors demonstrate **agentic properties**: the system not only generates text but **takes initiative to evaluate, revise, and decide** the next action based on intermediate results—without user intervention.
