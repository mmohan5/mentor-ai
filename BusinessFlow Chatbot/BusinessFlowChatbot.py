import streamlit as st
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

st.set_page_config(layout="wide")

# -------------------- GPU CHECK --------------------
def gpu_check():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        if "No devices were found" in output:
            st.error("🚫 No GPU detected. Exiting.")
            st.stop()
        elif "%" in output:
            st.success("✅ GPU check passed.")
        else:
            st.warning("⚠️ GPU detected but not in use. Exiting.")
            st.stop()
    except subprocess.CalledProcessError:
        st.error("❌ Could not check GPU. Exiting.")
        st.stop()

gpu_check()

# -------------------- PROMPT MANAGEMENT --------------------
PROMPT_FILE = Path("prompts.yaml")

def load_prompts():
    with open(PROMPT_FILE, "r") as f:
        return yaml.safe_load(f)

def save_prompts(data):
    with open(PROMPT_FILE, "w") as f:
        yaml.safe_dump(data, f)

PROMPTS = load_prompts()
SECTIONS = list(PROMPTS.keys())

# -------------------- STATE & FLOW DEFINITIONS --------------------
llm = ChatOllama(model="llama3.1", device="cuda", temperature=0)

class BusinessPlanState(TypedDict):
    user_input: str
    responses: Dict[str, str]
    sections: List[str]
    current_section: int
    history: Dict[str, List[str]]

class BusinessPlanBuilder:
    def __init__(self):
        graph = StateGraph(BusinessPlanState)
        for section in SECTIONS:
            graph.add_node(section, self.ask_initial_question)
            graph.add_node(f"{section} Followup", self.ask_followup_question)
            next_node = SECTIONS[SECTIONS.index(section)+1] if section != "Executive Summary" else "Compile Plan"
            graph.add_edge(section, f"{section} Followup")
            graph.add_edge(f"{section} Followup", next_node)
            
        graph.add_node("Compile Plan", self.compile_plan)
        graph.add_edge("Compile Plan", END)
        graph.set_entry_point("Company Description")
        self.graph = graph.compile()

    def ask_initial_question(self, state: BusinessPlanState):
        section = state["sections"][state["current_section"]]
        question = PROMPTS[section]
        state["responses"][section] = state["user_input"]
        state["history"].setdefault(section, []).append(f"Q: {question}\nA: {state['user_input']}")
        st.session_state.phase = "await_followup"
        return state

    def ask_followup_question(self, state: BusinessPlanState):
        section = state["sections"][state["current_section"]]
        initial_response = state["responses"][section]
        question = PROMPTS[section]
        followup_prompt = f"""Analyze this business plan response and identify any missing details:
        
        Question: {question}
        Response: {initial_response}

        Create a follow-up question to fill any gaps and complete incomplete responses if the response does not answer all questions asked. Do not call it the follow-up question, mention "follow-up question at all", or use any kind of qualifiers or titles to label it as something similar. Only ask the question directly."""
        followup = llm.invoke(followup_prompt).content.strip()
        st.session_state.followup_question = followup

        if state["user_input"]:
            state["responses"][section] += f"\n\n{state['user_input']}"
            state["history"][section].append(f"Q: {followup}\nA: {state['user_input']}")
            state["current_section"] += 1
            st.session_state.phase = "initial"
        return state

    def compile_plan(self, state: BusinessPlanState):
        all_qa = "\n\n".join(["\n".join(qas) for qas in state["history"].values()])
        prompt = f"""Based on the following structured questions and answers, generate a detailed, professional, and well-formatted business plan. Ensure that it follows this structure:

        1. Executive Summary
        2. Company Description
        3. Market Analysis
        4. Organization and Management
        5. Service or Product Line
        6. Marketing and Sales
        7. Funding Request
        8. Financial Projections

        Here is the collected information:

        {all_qa}

        Do not fabricate any information or make assumptions. Make sure to only use the provided information to generate the business plan.
        Now, craft a professional business plan with clear, concise, and structured content.
        """
        plan = llm.invoke(prompt).content.strip()
        disclaimer = "\n\n\n📌 NOTE: The generated business plan is a starting point and may require further refinement and correction."
        plan += disclaimer
        state["responses"]["Final Plan"] = plan
        st.session_state.final_plan = plan
        return state

# -------------------- SESSION STATE --------------------
def init_state():
    st.session_state.setdefault("builder", BusinessPlanBuilder())
    st.session_state.setdefault("state", {
        "user_input": "",
        "responses": {},
        "sections": SECTIONS,
        "current_section": 0,
        "history": {}
    })
    st.session_state.setdefault("phase", "initial")
    st.session_state.setdefault("waiting_for_llm", False)
    st.session_state.setdefault("llm_ready_for_input", False)
    st.session_state.setdefault("pending_input", None)
    st.session_state.setdefault("followup_question", "")
    st.session_state.setdefault("final_plan", "")

init_state()

# -------------------- MAIN INTERFACE --------------------
st.sidebar.image("charlotte_logo.png", width=120)
page = st.sidebar.radio("Navigate", ["User", "Admin"])

if page == "User":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📘 Instructions")
    st.sidebar.markdown("""
    Answer each question to the best of your ability.
                
    Here are some commands you can use:
    - Type **skip** to move ahead
    - Type **back** to go to the previous section
    - Type **restart** to start over
    
    ⚠️ Warning: **back** and **restart** will erase answers!
    """)


    st.title("💬 Business Plan Chatbot")

    # Render chat history with section headers
    for section in SECTIONS:
        for qa in st.session_state["state"]["history"].get(section, []):
            q, a = qa.split("\nA: ")
            st.chat_message("assistant").markdown(q.replace("Q: ", ""))
            st.chat_message("user").markdown(a)

    # Final Plan
    if st.session_state.final_plan:
        st.success("✅ Your complete business plan:")
        st.markdown(st.session_state.final_plan)
    else:
        if st.session_state.state["current_section"] >= len(st.session_state.state["sections"]):
            st.session_state.builder.compile_plan(st.session_state.state)
            st.rerun()
        
        current_section = st.session_state.state["sections"][st.session_state.state["current_section"]]
        phase = st.session_state.phase
        prompt = PROMPTS[current_section] if phase == "initial" else st.session_state.followup_question
        st.chat_message("assistant").markdown(f"**{current_section}**\n\n{prompt}")

        # First-time init rerun
        if not st.session_state.llm_ready_for_input:
            st.session_state.llm_ready_for_input = True
            st.rerun()

        # 🧠 Phase 1: Generating Response (before LLM runs)
        if st.session_state.waiting_for_llm and st.session_state.pending_input:
            with st.chat_message("assistant"):
                st.markdown("⏳ _Generating your response..._")


        # 🧠 Phase 2: Actually run LLM logic
        if st.session_state.waiting_for_llm and st.session_state.pending_input is not None:
            st.session_state.state["user_input"] = st.session_state.pending_input
            st.session_state.pending_input = None  # only run once

            if st.session_state.phase == "initial":
                st.session_state.builder.ask_initial_question(st.session_state.state)
                st.session_state.state["user_input"] = ""
                st.session_state.builder.ask_followup_question(st.session_state.state)
            elif st.session_state.phase == "await_followup":
                st.session_state.builder.ask_followup_question(st.session_state.state)

            st.session_state.waiting_for_llm = False
            st.rerun()

        
        user_input = st.chat_input("Type your answer here...")


        if user_input:
            if user_input.lower() == "skip":
                current_section_idx = st.session_state.state["current_section"]

                if current_section_idx < len(st.session_state.state["sections"]):
                    section = st.session_state.state["sections"][current_section_idx]
                    question = PROMPTS[section]

                    if st.session_state.phase == "initial":
                        st.session_state.state["responses"][section] = "Skipped"
                        st.session_state.state["history"].setdefault(section, []).append(f"Q: {question}\nA: Skipped")

                        st.session_state.state["history"][section].append("Q: [Follow-up question skipped due to initial skip]\nA: Skipped")
                        st.session_state.state["current_section"] += 1
                        st.session_state.phase = "initial"  # reset for next section

                    # If skipping during the follow-up phase only
                    elif st.session_state.phase == "await_followup":
                        followup_q = st.session_state.followup_question or "[Follow-up question]"
                        st.session_state.state["history"][section].append(f"Q: {followup_q}\nA: Skipped")
                        st.session_state.state["current_section"] += 1
                        st.session_state.phase = "initial"

                st.rerun()

            elif user_input.lower() == "back":
                st.session_state.state["current_section"] = max(0, st.session_state.state["current_section"] - 1)
                st.rerun()
            elif user_input.lower() == "restart":
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            else:
                st.session_state.pending_input = user_input
                st.session_state.waiting_for_llm = True
                st.rerun()
                




# -------------------- ADMIN PANEL --------------------
if page == "Admin":
    st.title("🛠️ Admin: Edit Prompts")
    updated = {}
    for section, question in PROMPTS.items():
        updated[section] = st.text_area(section, value=question)
    if st.button("💾 Save Changes"):
        save_prompts(updated)
        st.success("Prompts saved. Reload the app to apply changes.")
