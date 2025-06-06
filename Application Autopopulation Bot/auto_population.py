import streamlit as st
# from langchain_community.chat_models import ChatOllama
from fpdf import FPDF
import subprocess
import sys
import time

# List of predefined questions that the AI will answer
questions = [
    # Business Details
    "What is the elevator pitch?",
    "What problem is the company solving?",
    "What is the company's solution to the problem they are solving?",
    "How is the company's solution defensible in the marketplace?",
    "What are the patent details?",
    "What risks does the company face?",
    "How did the company discover their customers?",
    "What is the customer description?",
    "What is the customer acquisition strategy?",
    "What is the company's revenue model?",
    "What is the market opportunity for the company?",
    "What is the company's competitive landscape?",
]

def main():
    # Ensure GPU is available before continuing
    gpu_check()

    st.title("Seed Grant Application Assistant")

    # Initialize session state to persist data across Streamlit reruns
    if 'company_description' not in st.session_state:
        st.session_state.company_description = ""
    if 'answers' not in st.session_state:
        st.session_state.answers = ["" for _ in questions]
    if 'selected_sidebar_button' not in st.session_state:
        st.session_state.selected_sidebar_button = "Enter Company Description"
    if 'generating_answers' not in st.session_state:
        st.session_state.generating_answers = False
    if 'show_confirmation' not in st.session_state:
        st.session_state.show_confirmation = False
    if 'success_message' not in st.session_state:
        st.session_state.success_message = ""

    # Sidebar navigation
    st.sidebar.image("charlotte_logo.png", width=120)
    st.sidebar.title("Navigation")
    
    options = ["Enter Company Description", "View/Edit Answers"]
    
    def update_sidebar_selection():
        st.session_state.selected_sidebar_button = st.session_state.sidebar_radio

    # Sidebar radio for selecting between description and answer view
    selected_sidebar_button = st.sidebar.radio(
        "Select a page", 
        options,
        on_change=update_sidebar_selection,
        disabled=st.session_state.generating_answers,
        key="sidebar_radio"
    )

    # Page 1: Text area to enter the company description
    if st.session_state.selected_sidebar_button == "Enter Company Description":
        company_description = st.text_area(
            "Enter your company description:",
            value=st.session_state.company_description,
            key="company_description",
            disabled=st.session_state.show_confirmation
        )

        # Save updated description in session state
        if company_description != st.session_state.company_description and company_description is not None:
            st.session_state.company_description = company_description

        # Trigger generation or confirmation modal
        if st.button("Generate Answers", disabled=st.session_state.generating_answers):
            if any(st.session_state.answers):  # If answers already exist, ask for confirmation
                st.session_state.show_confirmation = True
                st.rerun()
            else:
                st.session_state.generating_answers = True
                st.rerun()

        # Confirmation modal to warn user before overwriting existing answers
        if st.session_state.show_confirmation:
            st.warning("This action will replace all existing answers. Are you sure you want to proceed?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, proceed"):
                    st.session_state.generating_answers = True
                    st.session_state.show_confirmation = False
                    st.rerun()
            with col2:
                if st.button("No, cancel"):
                    st.session_state.show_confirmation = False
                    st.rerun()

    # Page 2: View or edit each answer
    elif st.session_state.selected_sidebar_button == "View/Edit Answers":
        for i, question in enumerate(questions):
            answer = st.text_area(
                question,
                value=st.session_state.answers[i],
                key=f"answer_{i}"
            )

            if answer != st.session_state.answers[i]:
                st.session_state.answers[i] = answer

        # Save answers and generate downloadable PDF
        if st.button("Save Answers"):
            st.session_state.success_message = "Answers saved successfully!"
            pdf_content = create_pdf(questions, st.session_state.answers)
            st.download_button(
                label="Download PDF",
                data=pdf_content,
                file_name="seed_grant_application.pdf",
                mime="application/pdf"
            )

    # AI generation logic triggered
    if st.session_state.generating_answers:
        with st.spinner("Generating answers... This may take up to a few minutes."):
            st.session_state.answers = generate_answers(st.session_state.company_description, questions)
        st.session_state.generating_answers = False
        st.session_state.success_message = "Answers generated successfully!"
        st.rerun()

    # Show success message and clear it after display
    if st.session_state.success_message:
        st.success(st.session_state.success_message)
        st.session_state.success_message = ""

# Verifies if GPU is available using nvidia-smi
def gpu_check():
    if 'gpu_message_shown' not in st.session_state:
        st.session_state.gpu_message_shown = False

    if not st.session_state.gpu_message_shown:
        try:
            output = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            if "No devices were found" in output:
                st.error("No GPU detected. Exiting.")
                sys.exit(1)
            elif "%" in output:
                st.session_state.gpu_message_shown = True
                success_placeholder = st.empty()
                success_placeholder.success("GPU check passed.", icon="✅")
                time.sleep(1)
                success_placeholder.empty()
            else:
                st.warning("GPU is detected but not being used. Exiting.")
                sys.exit(1)
        except subprocess.CalledProcessError:
            st.error("Unable to check GPU status. Exiting.")
            sys.exit(1)

# Generate a PDF with the final answers
def create_pdf(questions, answers):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Seed Grant Application", ln=True, align='C')
    pdf.ln(10)
    
    # List each question and answer pair
    pdf.set_font("Arial", size=12)
    for question, answer in zip(questions, answers):
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, txt=question)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=answer)
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')


# AI model and hallucination validation setup
from langchain_community.chat_models import ChatOllama
from transformers import pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings

nltk.download('punkt_tab')  # Tokenizer used for splitting text

# Splits text into manageable token-sized chunks
def chunk_text(text, max_tokens=80, overlap=8):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        sentence_tokens = len(words)
        
        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + " "
            current_chunk_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_chunk_tokens = sentence_tokens
            
            while current_chunk_tokens > max_tokens:
                if current_chunk_tokens <= max_tokens + max_tokens/2.1:
                   break

                words = word_tokenize(current_chunk)
                chunk = " ".join(words[:max_tokens])
                chunks.append(chunk)
                current_chunk = " ".join(words[max_tokens-overlap:]) + " "
                current_chunk_tokens = len(words[max_tokens-overlap:])

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Uses NLI model to check if chunk is grounded in the description
def check_hallucination(nli_model, chunk, description):
    result = nli_model(chunk, [description], hypothesis_template="This text is true: {}")
    return result['scores'][0] < 0.81  # Low score indicates a possible hallucination

# Rewrite answer using LLM if hallucinated chunks were found
def regenerate_answer(llm, chunks, description, question, answer):
    prompt = f"""
    Company Description: ""{description}""

    Question: ""{question}""

    Here is the previous answer:
    ""{answer}""

    The following chunks of text from the previous answer may contain hallucinations:
    ""{chunks}""

    Please rewrite the answer without any hallucinations, ensuring it is firmly grounded in the company description provided. If you cannot rewrite it accurately based on the given information, respond with 'Information not provided'.

    REQUIREMENTS:
    - Be concise and relevant.
    - Do not include any disclaimers or self-references.
    - Act as though you are the company representative trying to inform about your company.
    - Provide only plain text without formatting or special characters.
    - Be descriptive and provide concrete detail, but only if it's supported by the company description.
    - VERY IMPORTANT: Be sure to rewrite or omit the chunks marked as hallucinations! For rewritten chunks, ensure that the answer is firmly grounded in the company description.
    """
    return llm.invoke(prompt).content

# Main logic to generate and validate answers
def generate_answers(description, questions):
    llm = ChatOllama(model="llama3.1", device="cuda", temperature=0)
    nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
    no_info = "Information not found"
    description_chunks = chunk_text(description, 600, 10)
    answers = []

    try:
        for question in questions:
            # Prompt LLM to generate answer based solely on description
            prompt = f"""
            Company Description: ""{description}""

            Question: ""{question}""

            FOLLOW THESE REQUIREMENTS:
            - Please provide a concise and relevant answer to the question based on the company description as if you are the company representative answering it.
            - Do not say you are 'attempting' to answer the question or provide any other disclaimers.
            - Do not make any references to yourself or use 'I', 'us', 'we', or any personal pronouns.
            - Use accurate and precise language and information based on the company description.
            - If you do not have enough information to answer the question, output 'Information not provided'.
            - If you are not sure about specific technical details, avoid making them up or mentioning them.
            - If you lack enough details that you cannot provide an answer firmly based in the company description, output 'Information not provided'.
            - Act as though you are the company representative trying to inform about your company.
            - Everything should be in plain text. Do not include any formatting or special characters.
            - Be descriptive and provide concrete detail.
"""
            answer = llm.invoke(prompt).content

            # Check for hallucinated chunks and attempt to correct up to 2 times
            for attempts in range(3):
                chunks = chunk_text(answer)
                bad_chunks = []

                for chunk in chunks:
                    bad_one = True
                    for description_chunk in description_chunks:
                        if not check_hallucination(nli_model, chunk, description_chunk):
                            bad_one = False
                            break
                    if bad_one:
                        bad_chunks.append(chunk)

                if not bad_chunks:
                    break  # All content is grounded
                elif attempts == 2:
                    answer = no_info  # Fallback if unable to fix hallucinations
                    break
                else:
                    # Regenerate using only grounded content
                    answer = regenerate_answer(llm, "\n".join(bad_chunks), description, question, answer)

            answers.append(answer)

    except Exception as e:
        print(f"Error generating answers: {e}")
        answers = ["Error generating answer" for _ in questions]

    return answers


if __name__ == "__main__":
    main()
