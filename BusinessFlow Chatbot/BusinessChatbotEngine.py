import yaml
import os
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List
import asyncio
# To use Anthropic's Claude model, uncomment the following lines:
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv

# load_dotenv()

# ---- TypedDict defining the structure of the state used in the business plan process
class BusinessPlanState(TypedDict):    
    going_back: bool                      # Whether the user typed "back" to return to a previous section
    responses: Dict[str, str]             # Stores all the user's responses by section name
    sections: List[str]                   # Ordered list of section names
    current_section: int                  # Index of the current section being asked
    history: Dict[str, List[str]]         # A list of Q&A history for each section


class BusinessPlanBuilder:
    def __setup(self):
        # Initialize the language model (ChatOllama using llama3.1 running on CUDA)
        self.llm = ChatOllama(model="llama3.1", device="cuda", temperature=0)
        # self.llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))


        # Load prompt templates from prompts.yaml
        def load_custom_prompts():
            current_dir = ""
            yaml_path = os.path.join(current_dir, "prompts.yaml")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return data.get("customs", {})

        self.customs = load_custom_prompts()

        # Extract section names and their associated prompt questions
        self.SECTIONS = [s["name"] for s in self.customs["sections"]]
        self.QUESTIONS = {s["name"]: s["prompt"] for s in self.customs["sections"]}

        # Async coordination primitives for input/output synchronization
        self.allow_input_condition = asyncio.Condition()
        self.input_processed_condition = asyncio.Condition()
        self.output_ready_condition = asyncio.Condition()

    def __init__(self):
        self.__setup()

        # Define the state graph (LangGraph) workflow
        graph = StateGraph(BusinessPlanState)
        
        # Define each logical step (node) in the workflow
        graph.add_node("Ask Initial Question", self.__ask_initial_question)
        graph.add_node("Ask Followup Question", self.__ask_followup_question)
        graph.add_node("Compile Plan", self.__compile_business_plan)

        # Define the path through the graph
        graph.add_edge("Ask Initial Question", "Ask Followup Question")
        graph.add_conditional_edges(
            "Ask Followup Question", self.__route_next,
            {
                "Ask Initial Question": "Ask Initial Question",
                "Compile Plan": "Compile Plan",
                END: END
            }
        )
        graph.add_edge("Compile Plan", END)
        graph.set_entry_point("Ask Initial Question")

        # Compile graph for execution
        self.graph = graph.compile()

        # Internal control flags
        self.user_input = ""
        self.output = ""
        self.allow_input = False
        self.is_output_ready = False
        self.last_served_output = ""

    # Starts execution of the graph with initial state
    async def invoke(self):
        return await self.graph.ainvoke({
            "going_back": False,
            "responses": {},
            "sections": self.SECTIONS,
            "current_section": 0,
            "history": {},
        }, {"recursion_limit": 1000})

    # Called by the backend when user input is received from the frontend
    async def set_user_input(self, user_input: str):
        if self.allow_input:
            async with self.allow_input_condition:
                self.user_input = user_input
                self.allow_input_condition.notify()

    # Waits for user input to arrive; handles timeout and notifies that input has been processed
    async def __wait_for_input(self, state: BusinessPlanState):
        self.allow_input = True

        async with self.allow_input_condition:
            try:
                await asyncio.wait_for(
                    self.allow_input_condition.wait_for(lambda: self.user_input != ""),
                    timeout=3600  # 1 hour timeout
                )
            except asyncio.TimeoutError:
                # Handle timeout by exiting the session
                self.allow_input = False
                self.output = "Timeout: No input received within 60 minutes."
                state["responses"]["Final Plan"] = "Timed out due to inactivity."
                self.user_input = "exit"
                return state

        # After input is received, signal the backend that processing is complete
        async with self.input_processed_condition:
            self.is_output_ready = False
            self.allow_input = False
            self.input_processed_condition.notify_all()

        # Strip input to remove whitespace
        self.user_input = self.user_input.strip()
        return state

    # Handles a full user interaction step (ask → wait → process)
    async def __input_handler(self, state: BusinessPlanState, section_name: str, question: str, is_followup_question: bool = False):
        self.user_input = ""

        # Notify frontend that output is ready
        async with self.output_ready_condition:
            self.is_output_ready = True
            self.output_ready_condition.notify_all()

        # Wait for user input
        state = await self.__wait_for_input(state)

        # Handle special commands from user
        if self.user_input.lower() in ["exit", "back", "skip", "restart"]:
            self.user_input = self.user_input.lower()

        # Ensure history exists
        if section_name not in state["history"]:
            state["history"][section_name] = []

        # Handle control commands
        if self.user_input == "exit":
            return state
        elif self.user_input == "back":
            state["history"][section_name] = []
            state["responses"][section_name] = []
            if not is_followup_question:
                state["going_back"] = True
                state["current_section"] = max(state["current_section"] - 1, 0)
            return state
        elif self.user_input == "restart":
            state["history"].clear()
            state["responses"].clear()
            state["current_section"] = 0
            state["going_back"] = True
            return state
        elif self.user_input == "skip":
            if not is_followup_question:
                state["responses"][section_name] = "Skipped."
                state["history"][section_name].append(f"Q: {question}\nA: Skipped.")
                state["history"][section_name].append("Q: Followup question not generated.\nA: Skipped.")
            else:
                state["responses"][section_name] += "\n\nSkipped."
                state["history"][section_name].append(f"Q: {question}\nA: Skipped.")
            state["current_section"] += 1
            return state

        # Normal input
        if not is_followup_question:
            state["responses"][section_name] = self.user_input
            state["history"][section_name].append(f"Q: {question}\nA: {self.user_input}")
        else:
            state["responses"][section_name] += f"\n\n{self.user_input}"
            state["history"][section_name].append(f"Q: {question}\nA: {self.user_input}")
            state["current_section"] += 1

        return state

    # First question for each section
    async def __ask_initial_question(self, state: BusinessPlanState):
        state["going_back"] = False

        if state["current_section"] >= len(state["sections"]):
            return state

        section = state["sections"][state["current_section"]]
        question = self.QUESTIONS[section]
        self.output = f"**{section}** - \n{question}"

        state = await self.__input_handler(state, section, question)
        return state

    # Follow-up question generated based on first answer
    async def __ask_followup_question(self, state: BusinessPlanState):
        if (
            state["going_back"]
            or self.user_input in ["skip", "exit"]
            or state["current_section"] >= len(state["sections"])
        ):
            return state

        section = state["sections"][state["current_section"]]
        question = self.QUESTIONS[section]
        initial_response = state["responses"][section]

        # Generate follow-up question using prompt template
        followup_prompt = self.customs["followup_prompt"].format(
            question=question, response=initial_response
        )
        followup_question = self.llm.invoke(followup_prompt).content.strip()

        self.output = f"**{section}** - \n{followup_question}"
        state = await self.__input_handler(state, section, question, is_followup_question=True)
        return state

    # Determines what to do next in the graph after follow-up
    def __route_next(self, state: BusinessPlanState) -> str:
        if self.user_input == "exit":
            return END
        elif state["current_section"] >= len(state["sections"]):
            return "Compile Plan"
        else:
            return "Ask Initial Question"

    # Final step: compile a business plan from all previous Q&A
    async def __compile_business_plan(self, state: BusinessPlanState):
        full_qa = "\n\n".join(["\n".join(qas) for qas in state["history"].values()])
        prompt = self.customs["compile_plan_prompt"].format(all_qa=full_qa)

        # Generate the final plan
        refined_business_plan = self.llm.invoke(prompt).content.strip()

        disclaimer = (
            "\n\n📌 PLEASE NOTE: The generated business plan is a starting point and may require further refinement and correction."
        )
        refined_business_plan += disclaimer

        # Save to state and output
        state["responses"]["Final Plan"] = refined_business_plan
        self.output = f"\n--- Your Complete Business Plan ---\n\n{refined_business_plan}"

        async with self.output_ready_condition:
            self.is_output_ready = True
            self.output_ready_condition.notify_all()

        return state
