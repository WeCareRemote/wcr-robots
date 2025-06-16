"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Annotated, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from robot_icebreaker_agent.custom_prompt import ICEBREAKER_PROMPT, question_json

load_dotenv()


class Configuration(BaseModel):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    system_prompt: str = Field(
        default=ICEBREAKER_PROMPT,
        description="System prompt for the agent. ",
        json_schema_extra={
            "langgraph_nodes": ["call_model"],
            "langgraph_type": "prompt",
        },
    )

    model: Annotated[
        Literal[
            "anthropic/claude-3-7-sonnet-latest",
            "anthropic/claude-3-5-haiku-latest",
            "openai/o1",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "openai/o3-mini",
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="gpt-4o-mini",
        description="The name of the language model to use for the agent's main interactions. "
                    "Should be in the form: provider/model-name.",
        json_schema_extra={"langgraph_nodes": ["call_model"]},
    )


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    llm = ChatOpenAI(
        model=configuration.get("model", "gpt-4o-mini"),
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    user_prompt = HumanMessage(content=state.changeme)
    system_prompt = configuration.get("system_prompt", "You are a helpful AI assistant.")
    system_prompt += f"\n\n Here are the question which you ask the user:\n {question_json}"
    message = [configuration.get("system_prompt", "You are a helpful AI assisstant."), user_prompt]

    response = await llm.ainvoke(message)

    return {
        "changeme": response.content
    }


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
