import streamlit as st
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

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
    st.title("Seed Grant Application Assistant")

    # Initialize session state variables if not already initialized
    if 'company_description' not in st.session_state:
        st.session_state.company_description = ""
    if 'answers' not in st.session_state:
        st.session_state.answers = ["" for _ in questions]
    if 'selected_sidebar_button' not in st.session_state:
        st.session_state.selected_sidebar_button = "Enter Company Description"

    # Sidebar to choose between options
    st.sidebar.title("Navigation")
    selected_sidebar_button = st.sidebar.radio(
        "Select a page", 
        ["Enter Company Description", "View/Edit Answers"],
        index=["Enter Company Description", "View/Edit Answers"].index(st.session_state.selected_sidebar_button)
    )

    st.session_state.selected_sidebar_button = selected_sidebar_button

    # Page for company description input
    if selected_sidebar_button == "Enter Company Description":
        company_description = st.session_state.company_description
        st.text_area(
            "Enter your company description:", 
            value=company_description,
            key="company_description"  # Unique key to prevent re-renders clearing text
        )

        # Update session state only when user changes the text area content
        if company_description != st.session_state.company_description:
            st.session_state.company_description = company_description

        if st.button("Submit & Generate Answers"):
            # st.session_state.company_description = company_description
            st.success("Description submitted successfully! Wait for the answers to be generated...")
            st.session_state.answers = generate_answers(st.session_state.company_description, questions)
            st.success("Answers generated successfully!")

    # Page for displaying and editing answers
    elif selected_sidebar_button == "View/Edit Answers":
        for i, question in enumerate(questions):
            answer = st.text_area(
                question,
                value=st.session_state.answers[i],
                key=f"answer_{i}"  # Unique key for each answer field
            )

            # Update session state if the user edits the answer
            if answer != st.session_state.answers[i]:
                st.session_state.answers[i] = answer

        if st.button("Save Answers"):
            st.success("Answers saved successfully!")
            pdf_content = create_pdf(questions, st.session_state.answers)
            st.download_button(
                label="Download PDF",
                data=pdf_content,
                file_name="seed_grant_application.pdf",
                mime="application/pdf"
            )

def create_pdf(questions, answers):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Seed Grant Application", ln=True, align='C')
    pdf.ln(10)
    
    # Add questions and answers
    pdf.set_font("Arial", size=12)
    for question, answer in zip(questions, answers):
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, txt=question)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=answer)
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')


def generate_answers(description, questions):
    llm = ChatOllama(model="llama3.2", device="cuda", temperature=0)
    answers = []

    try:
        for question in questions:
            prompt = f"""
            Company Description: {description}

            Question: {question}

            Please provide a concise and relevant answer to the question based on the company description as if you are the company representative answering it. Do not say you are 'attempting' to answer the question or provide any other disclaimers. Simply provide the answer as if you are the company representative:
            """
            answer = llm.invoke(prompt)
            answers.append(answer.content)
    except Exception as e:
        st.error(f"Error generating answers: {e}")
        answers = ["Error generating answer" for _ in questions]

    return answers

if __name__ == "__main__":
    main()
