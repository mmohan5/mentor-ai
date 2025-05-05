import streamlit as st
import requests
import yaml
from pathlib import Path
import subprocess
import time

st.set_page_config(layout="wide")

API_URL = "http://localhost:8000"

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

# -------------------- SESSION STATE --------------------
def init_state():
    if "session_id" not in st.session_state:
        try:
            res = requests.post(f"{API_URL}/start")
            res.raise_for_status()
            data = res.json()

            session_id = data.get("session_id")
            output = data.get("output", "")
            allow_input = data.get("allow_input", False)

            if not session_id:
                st.error("⚠️ Failed to get a valid session ID from backend.")
                st.stop()

            # Wait and keep polling if output isn't ready yet
            max_wait_seconds = 20
            poll_interval = 0.5
            waited = 0

            while not output and waited < max_wait_seconds:
                time.sleep(poll_interval)
                waited += poll_interval

                # Try polling again (you could also create a dedicated `/status` endpoint for robustness)
                res = requests.post(f"{API_URL}/start")
                res.raise_for_status()
                data = res.json()
                output = data.get("output", "")

            if not output:
                st.error("⚠️ Backend timed out waiting for initial output.")
                st.stop()

            st.session_state.session_id = session_id
            st.session_state.history = [("assistant", output)]
            st.session_state.allow_input = allow_input
            st.session_state.awaiting_llm = False
            st.session_state.user_input = ""
            st.session_state.finished = "--- Your Complete Business Plan ---" in output

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error initializing session: {e}")
            st.stop()

init_state()

# -------------------- USER PAGE --------------------
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

    for key, default in {
        "history": [],
        "awaiting_llm": False,
        "user_input": "",
        "finished": False,
        "allow_input": False,
        "session_id": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Initialize session (start server and get first output)
    if st.session_state.session_id is None:
        with st.spinner("⏳ Starting session..."):
            res = requests.post(f"{API_URL}/start")
            res.raise_for_status()
            data = res.json()

            st.session_state.session_id = data["session_id"]
            output = data["output"]
            st.session_state.allow_input = data["allow_input"]

            st.session_state.history.append(("assistant", output))

            if "--- Your Complete Business Plan ---" in output:
                st.session_state.finished = True

            st.rerun()

    # Display conversation history
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    # Handle restart
    if st.session_state.get("restart_requested", False):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # Handle user input
    if st.session_state.allow_input and not st.session_state.finished:
        user_input = st.chat_input("Type your answer here...")

        if user_input:
            if user_input.lower() == "restart":
                st.session_state.restart_requested = True
                st.rerun()

            st.session_state.history.append(("user", user_input))
            st.session_state.awaiting_llm = True
            st.session_state.user_input = user_input
            st.session_state.allow_input = False
            st.rerun()

    # Handle awaiting LLM reply
    if st.session_state.awaiting_llm and not st.session_state.finished:
        with st.spinner("⏳ Generating..."):
            payload = {
                "session_id": st.session_state.session_id,
                "user_input": st.session_state.user_input
            }
            res = requests.post(f"{API_URL}/step", json=payload)
            res.raise_for_status()
            data = res.json()

            output = data["output"]
            st.session_state.allow_input = data["allow_input"]
            st.session_state.awaiting_llm = False
            st.session_state.user_input = ""

            st.session_state.history.append(("assistant", output))

            with st.chat_message("assistant"):
                st.markdown(output)

            if "--- Your Complete Business Plan ---" in output:
                st.session_state.finished = True

            st.rerun()

    # Plan complete
    if st.session_state.finished:
        st.success("✅ Business Plan Complete!")


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