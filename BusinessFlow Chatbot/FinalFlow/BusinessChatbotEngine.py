import yaml
import os
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List

import asyncio



class BusinessPlanState(TypedDict):    
    going_back: bool
    responses: Dict[str, str]
    sections: List[str]
    current_section: int
    history: Dict[str, List[str]]

class BusinessPlanBuilder:
    def __setup(self):
        self.llm = ChatOllama(model="llama3.1", device="cuda", temperature=0)

        def load_custom_prompts():
            current_dir = ""
            yaml_path = os.path.join(current_dir, "prompts.yaml")
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            return data.get("customs", {})

        self.customs = load_custom_prompts()

        self.SECTIONS = [s["name"] for s in self.customs["sections"]]
        self.QUESTIONS = {s["name"]: s["prompt"] for s in self.customs["sections"]}
        
        self.condition = asyncio.Condition()

    def __init__(self):
        self.__setup()
        graph = StateGraph(BusinessPlanState)
        
        initial_question_node = "Ask Initial Question"
        followup_node = "Ask Followup Question"
        compile_node = "Compile Plan"

        graph.add_node(initial_question_node, self.__ask_initial_question)
        graph.add_node(followup_node, self.__ask_followup_question)
        graph.add_node(compile_node, self.__compile_business_plan)

        graph.add_edge(initial_question_node, followup_node)
        graph.add_conditional_edges(followup_node, self.__route_next,
                                    {
                                        initial_question_node: initial_question_node,
                                        compile_node: compile_node,
                                        END: END
                                    }
                                    )
        graph.add_edge(compile_node, END)

        graph.set_entry_point(initial_question_node)
        self.graph = graph.compile()


        self.user_input = ""
        self.output = ""
        self.allow_input = False
        self.last_served_output = ""


    async def invoke(self):
        return await self.graph.ainvoke({
            "going_back": False,
            "responses": {},
            "sections": self.SECTIONS,
            "current_section": 0,
            "history": {},
        }, {"recursion_limit": 1000})

    async def set_user_input(self, user_input: str):
        if self.allow_input:
            async with self.condition:
                self.user_input = user_input
                self.condition.notify()

    async def __wait_for_input(self, state: BusinessPlanState):
        self.allow_input = True

        async with self.condition:
            try:
                await asyncio.wait_for(
                    self.condition.wait_for(lambda: self.user_input != ""),
                    timeout=3600
                )
            except asyncio.TimeoutError:
                self.allow_input = False
                self.output = "Timeout: No input received within 60 minutes."
                state["responses"]["Final Plan"] = "Timed out due to inactivity."
                self.user_input = "exit"
                return state

        self.allow_input = False
        self.user_input = self.user_input.strip()
        return state

    async def __input_handler(self, state: BusinessPlanState, section_name: str, question: str, is_followup_question: bool = False):
        # print("Section Num: ", state["current_section"])
        self.user_input = ""
        state = await self.__wait_for_input(state)

        if self.user_input.lower() in ["exit", "back", "skip", "restart"]:
            self.user_input = self.user_input.lower()

        if section_name not in state["history"]:
            state["history"][section_name] = []

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
                state["responses"][section_name] += "\n\nSkipped."
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
        else:  # follow-up
            state["responses"][section_name] += f"\n\n{self.user_input}"
            state["history"][section_name].append(f"Q: {question}\nA: {self.user_input}")
            state["current_section"] += 1
        return state



    async def __ask_initial_question(self, state: BusinessPlanState):
        state["going_back"] = False

        if state["current_section"] >= len(state["sections"]):
            return state

        section = state["sections"][state["current_section"]]
        question = self.QUESTIONS[section]

        self.output = f"**{section}** - \n{question}"
        state = await self.__input_handler(state, section, question)

        return state


    async def __ask_followup_question(self, state: BusinessPlanState):
        if state["going_back"] or self.user_input == "skip" or self.user_input == "exit" or state["current_section"] >= len(state["sections"]):
            return state

        section = state["sections"][state["current_section"]]

        question = self.QUESTIONS[section]
        initial_response = state["responses"][section]
        followup_prompt = self.customs["followup_prompt"].format(question=question, response=initial_response)
        followup_question = self.llm.invoke(followup_prompt).content.strip()

        self.output = f"**{section}** - \n{followup_question}"
        state = await self.__input_handler(state, section, question, True)

        return state

    def __route_next(self, state: BusinessPlanState) -> str:
        if self.user_input == "exit":
            return END
        elif state["current_section"] >= len(state["sections"]):
            return "Compile Plan"
        else:
            return "Ask Initial Question"

    def __compile_business_plan(self, state: BusinessPlanState):
        full_qa = "\n\n".join(["\n".join(qas) for qas in state["history"].values()])
        prompt = self.customs["compile_plan_prompt"].format(all_qa=full_qa)

        refined_business_plan = self.llm.invoke(prompt).content.strip()
        disclaimer = "\n\n📌 PLEASE NOTE: The generated business plan is a starting point and may require further refinement and correction."
        refined_business_plan += disclaimer

        state["responses"]["Final Plan"] = refined_business_plan

        self.output = f"\n--- Your Complete Business Plan ---\n\n{refined_business_plan}"

        return state
