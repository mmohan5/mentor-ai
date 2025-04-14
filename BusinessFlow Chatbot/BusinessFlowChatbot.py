import streamlit as st
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
import time

st.set_page_config(layout="wide")

# -------------------- GPU CHECK --------------------
def gpu_check():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        
        if "No devices were found" in output:
            st.error("🚫 No GPU detected. Exiting.")
            st.stop()
        elif "%" in output:
            if "show_gpu_success" not in st.session_state:
                st.session_state.show_gpu_success = True
                st.rerun()
            elif st.session_state.show_gpu_success:
                st.success("✅ GPU check passed.")
                time.sleep(1)
                st.session_state.show_gpu_success = False
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
        raw = yaml.safe_load(f)

    customs = raw.get("customs", {})
    custom_sections = {s["name"]: s["prompt"] for s in customs.get("sections", [])}

    prompts = {
        "followup_prompt": customs.get("followup_prompt", ""),
        "compile_plan_prompt": customs.get("compile_plan_prompt", ""),
        "sections": list(custom_sections.keys()),
        "section_prompts": custom_sections
    }

    return prompts, raw

def save_prompts(data):
    with open(PROMPT_FILE, "w") as f:
        yaml.safe_dump(data, f)

PROMPTS, RAW_PROMPT_DATA = load_prompts()
SECTIONS = PROMPTS["sections"]

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

            is_last = SECTIONS.index(section) == len(SECTIONS) - 1
            next_node = "Compile Plan" if is_last else SECTIONS[SECTIONS.index(section)+1]

            graph.add_edge(section, f"{section} Followup")
            graph.add_edge(f"{section} Followup", next_node)

        graph.add_node("Compile Plan", self.compile_plan)
        graph.add_edge("Compile Plan", END)
        graph.set_entry_point(SECTIONS[0])  # dynamic entry point
        self.graph = graph.compile()


    def ask_initial_question(self, state: BusinessPlanState):
        section = state["sections"][state["current_section"]]
        question = PROMPTS["section_prompts"][section]
        state["responses"][section] = state["user_input"]
        state["history"].setdefault(section, []).append(f"Q: {question}\nA: {state['user_input']}")
        st.session_state.phase = "await_followup"
        return state

    def ask_followup_question(self, state: BusinessPlanState):
        section = state["sections"][state["current_section"]]
        initial_response = state["responses"][section]
        question = PROMPTS["section_prompts"][section]

        followup_template = PROMPTS["followup_prompt"]
        followup_prompt = followup_template.format(question=question, response=initial_response)

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
        prompt_template = PROMPTS["compile_plan_prompt"]
        prompt = prompt_template.format(all_qa=all_qa)

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
st.sidebar.image("charlotte_white_logo.png", width=160)
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

    for section in SECTIONS:
        for qa in st.session_state["state"]["history"].get(section, []):
            q, a = qa.split("\nA: ")
            st.chat_message("assistant").markdown(q.replace("Q: ", ""))
            st.chat_message("user").markdown(a)

    if st.session_state.final_plan:
        st.success("✅ Your complete business plan:")
        st.markdown(st.session_state.final_plan)
    else:
        if st.session_state.state["current_section"] >= len(st.session_state.state["sections"]):
            st.session_state.builder.compile_plan(st.session_state.state)
            st.rerun()
        
        current_section = st.session_state.state["sections"][st.session_state.state["current_section"]]
        phase = st.session_state.phase
        prompt = PROMPTS["section_prompts"][current_section] if phase == "initial" else st.session_state.followup_question
        st.chat_message("assistant").markdown(f"**{current_section}**\n\n{prompt}")

        if not st.session_state.llm_ready_for_input:
            st.session_state.llm_ready_for_input = True
            st.rerun()

        if st.session_state.waiting_for_llm and st.session_state.pending_input:
            with st.chat_message("assistant"):
                st.markdown("⏳ _Generating your response..._")

        if st.session_state.waiting_for_llm and st.session_state.pending_input is not None:
            st.session_state.state["user_input"] = st.session_state.pending_input
            st.session_state.pending_input = None

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
                    question = PROMPTS["section_prompts"][section]

                    if st.session_state.phase == "initial":
                        st.session_state.state["responses"][section] = "Skipped"
                        st.session_state.state["history"].setdefault(section, []).append(f"Q: {question}\nA: Skipped")
                        st.session_state.state["history"][section].append("Q: [Follow-up question skipped due to initial skip]\nA: Skipped")
                        st.session_state.state["current_section"] += 1
                        st.session_state.phase = "initial"
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
    - Press **Reset** to reset a prompt (this will erase unsaved changes)
    - Press **Save All Changes** to apply

    **Important Notes**:
    - `{question}` and `{response}` must appear in the follow-up prompt
    - `{all_qa}` must appear in the compile plan prompt
    """)

    customs = RAW_PROMPT_DATA.get("customs", {})
    defaults = RAW_PROMPT_DATA.get("defaults", {})

    # ---------- FOLLOW-UP PROMPT ----------
    st.markdown("### 🔁 Question Follow-Up Prompt")
    st.markdown("Must include `{question}` and `{response}`.\n")

    followup_custom = customs.get("followup_prompt", "")
    followup_default = defaults.get("followup_prompt", "")

    if "followup_prompt" not in st.session_state:
        st.session_state["followup_prompt"] = followup_custom

    if st.session_state.get("reset_followup_flag"):
        st.session_state["followup_prompt"] = followup_default
        st.session_state["reset_followup_flag"] = False
        st.rerun()

    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_area(
            "Follow-Up Prompt Template",
            value=st.session_state["followup_prompt"],
            key="followup_prompt",
            height=200,
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Reset", key="reset_followup"):
            st.session_state["reset_followup_flag"] = True
            st.rerun()

    st.markdown("---")

    # ---------- COMPILE PLAN PROMPT ----------
    st.markdown("### 🧠 Generate Plan Prompt")
    st.markdown("Must include `{all_qa}`.\n")

    compile_custom = customs.get("compile_plan_prompt", "")
    compile_default = defaults.get("compile_plan_prompt", "")

    if "compile_plan_prompt" not in st.session_state:
        st.session_state["compile_plan_prompt"] = compile_custom

    if st.session_state.get("reset_compile_flag"):
        st.session_state["compile_plan_prompt"] = compile_default
        st.session_state["reset_compile_flag"] = False
        st.rerun()

    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_area(
            "Compile Plan Prompt Template",
            value=st.session_state["compile_plan_prompt"],
            key="compile_plan_prompt",
            height=300,
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Reset", key="reset_compile"):
            st.session_state["reset_compile_flag"] = True
            st.rerun()

    st.markdown("---")

    # ---------- SECTION PROMPTS ----------
    st.markdown("### 📄 Plan Section Prompts")

    custom_section_map = {s["name"]: s["prompt"] for s in customs.get("sections", [])}
    default_section_map = {s["name"]: s["prompt"] for s in defaults.get("sections", [])}

    updated_sections = []

    for section in SECTIONS:
        default_prompt = default_section_map.get(section, "")
        session_key = f"section_prompt_{section}"
        reset_flag_key = f"reset_section_{section}"

        if session_key not in st.session_state:
            st.session_state[session_key] = custom_section_map.get(section, "")

        if st.session_state.get(reset_flag_key):
            st.session_state[session_key] = default_prompt
            st.session_state[reset_flag_key] = False
            st.rerun()

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{section}**")
        with col2:
            if st.button("Reset", key=f"reset_button_{section}"):
                st.session_state[reset_flag_key] = True
                st.rerun()

        st.text_area(
            "Prompt",
            value=st.session_state[session_key],
            key=session_key,
            height=100,
            label_visibility="collapsed"
        )

        st.markdown("\n--\n")

        updated_sections.append({
            "name": section,
            "prompt": st.session_state[session_key]
        })

    # ---------- SAVE CHANGES ----------
    if st.button("💾 Save Changes"):
        errors = []

        if "{question}" not in st.session_state["followup_prompt"] or "{response}" not in st.session_state["followup_prompt"]:
            errors.append("❌ Follow-up prompt must include `{question}` and `{response}`.")
        if "{all_qa}" not in st.session_state["compile_plan_prompt"]:
            errors.append("❌ Compile plan prompt must include `{all_qa}`.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            RAW_PROMPT_DATA["customs"]["followup_prompt"] = st.session_state["followup_prompt"]
            RAW_PROMPT_DATA["customs"]["compile_plan_prompt"] = st.session_state["compile_plan_prompt"]
            RAW_PROMPT_DATA["customs"]["sections"] = updated_sections
            save_prompts(RAW_PROMPT_DATA)
            st.success("✅ Custom prompts saved successfully.")
