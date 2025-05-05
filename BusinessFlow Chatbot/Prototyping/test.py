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

# gpu_check()

# -------------------- PROMPT MANAGEMENT --------------------
PROMPT_FILE = Path("prompts.yaml")

def load_prompts():
    with open(PROMPT_FILE, "r") as f:
        data = yaml.safe_load(f)
        sections = data.get("sections", [])
        prompts = {s["name"]: s["prompt"] for s in sections}
        return {
            "meta": data.get("__meta__", {}),
            "sections": sections,
            "prompts": prompts
        }

def save_prompts(data):
    with open(PROMPT_FILE, "w") as f:
        yaml.safe_dump(data, f)

PROMPT_DATA = load_prompts()
PROMPTS = PROMPT_DATA["prompts"]
SECTIONS = list(PROMPTS.keys())
META = PROMPT_DATA["meta"]

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
        self.section_nodes = {}

        for idx, section in enumerate(SECTIONS):
            init_node = f"section_{idx}"
            follow_node = f"{init_node}_followup"

            self.section_nodes[section] = (init_node, follow_node)

            graph.add_node(init_node, self.ask_initial_question)
            graph.add_node(follow_node, self.ask_followup_question)

            next_node = f"section_{idx+1}" if idx + 1 < len(SECTIONS) else "Compile Plan"
            graph.add_edge(init_node, follow_node)
            graph.add_edge(follow_node, next_node)

        graph.add_node("Compile Plan", self.compile_plan)
        graph.add_edge("Compile Plan", END)
        graph.set_entry_point("section_0")
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

        followup_template = META["followup_prompt"]
        followup_prompt = followup_template.format(
            question=question,
            response=initial_response
        )

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
        template = META["compile_plan_prompt"]

        prompt = template.format(all_qa=all_qa)

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

        # Generating Response (before LLM runs)
        if st.session_state.waiting_for_llm and st.session_state.pending_input:
            with st.chat_message("assistant"):
                st.markdown("⏳ _Generating your response..._")


        # Actually run LLM logic
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
    st.title("🛠️ Admin Panel")

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📘 Instructions")
    st.sidebar.markdown("""
    - Edit the **prompts** freely below
    - Use the **Reset** button restore a to the default (this will erase unsaved changes)
    - Press **Save All Changes** to apply

    **Important Notes**:
    - `{question}` and `{response}` must appear in the follow-up prompt
    - `{all_qa}` must appear in the compile plan prompt
    """)


    # Load current prompts
    with open(PROMPT_FILE, "r") as f:
        raw_data = yaml.safe_load(f)

    sections_data = raw_data.get("sections", [])
    meta_data = raw_data.get("__meta__", {})
    default_data = {
        "Executive Summary": "Summarize your business plan in a few sentences.",
        "Company Description": "Describe your business. What is its name, what problem does it solve, and what makes it unique?",
        "Market Analysis": "Who is your target audience, and who are your main competitors?",
        "Organization and Management": "What is your business structure, and who are the key members of your team?",
        "Service or Product Line": "What products or services do you offer, and how do they benefit customers?",
        "Marketing and Sales": "How will you attract and retain customers, and how will you generate sales?",
        "Funding Request": "How much funding do you need, and what will it be used for?",
        "Financial Projections": "What are your projected revenues, expenses, and break-even point?"
    }

    updated_sections = []
    st.markdown("### ✏️ Sections")
    for idx, section in enumerate(sections_data):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.text_input(f"Section Name {idx+1}", value=section["name"], key=f"name_{idx}")
            prompt = st.text_area(f"Prompt for '{name}'", value=section["prompt"], key=f"prompt_{idx}")
        with col2:
            if st.button(f"Reset {idx+1}", key=f"reset_{idx}"):
                if st.confirm(f"Reset section '{section['name']}' to default? Unsaved changes will be lost."):
                    section_name = section["name"]
                    if section_name in default_data:
                        st.session_state[f"prompt_{idx}"] = default_data[section_name]
                        st.session_state[f"name_{idx}"] = section_name

        updated_sections.append({
            "name": st.session_state[f"name_{idx}"],
            "prompt": st.session_state[f"prompt_{idx}"]
        })

    st.markdown("### 🔁 Follow-up Prompt Template")
    followup = st.text_area("Follow-up Prompt", value=meta_data.get("followup_prompt", ""), height=200)

    st.markdown("### 🧾 Compile Plan Prompt Template")
    compile_plan = st.text_area("Compile Plan Prompt", value=meta_data.get("compile_plan_prompt", ""), height=300)

    if st.button("💾 Save All Changes"):
        followup_valid = "{question}" in followup and "{response}" in followup
        compile_valid = "{all_qa}" in compile_plan

        if not followup_valid or not compile_valid:
            st.error("❌ Prompt templates must include required variables:")
            if not followup_valid:
                st.markdown("- Follow-up prompt must include `{question}` and `{response}`.")
            if not compile_valid:
                st.markdown("- Compile plan prompt must include `{all_qa}`.")
        else:
            new_data = {
                "__meta__": {
                    "followup_prompt": followup,
                    "compile_plan_prompt": compile_plan
                },
                "sections": updated_sections
            }
            with open(PROMPT_FILE, "w") as f:
                yaml.safe_dump(new_data, f, sort_keys=False)
            st.success("✅ Prompts saved successfully!")

