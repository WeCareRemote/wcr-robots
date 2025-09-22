from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, convert_to_messages, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Literal, Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from schema import Ask_Ai_AgentState
from core import get_model, settings

VERBOSE = True

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Access API keys and credentials
# OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
# GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
# # TIMESCALE_DB_URI  = os.environ["TIMESCALE_DB_URI"]
# # MAIN_AGENT_DB_URI = os.environ["MAIN_AGENT_DB_URI"]
# # TAVILY_API_KEY    = os.environ["TAVILY_API_KEY"]





#--------------------------------------- SCHEMAS ---------------------------------------

class GradeRelevance(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Is the user's message relevant to Refugee_Bridge support for using the wcr.is website or completing its forms (including greetings while using the service)? 'yes' or 'no'."
        )
    )

RELEVANCE_SYSTEM_MESSAGE = """
You are `ask ai`, an assistant for refugees using the wcr.is website. The site has multiple forms; users pick the one matching their current need and fill it out in another tab. While completing a form, they may ask about words, legal terms, or any confusing part. Your job is to guide them so they can finish the form correctly.

TASK: Decide if the user's message is relevant to Refugee_Bridge assistance. Return ONLY a binary score: 'yes' or 'no'.

Mark as 'yes' if the message concerns: using wcr.is; choosing/finding the right form; understanding or answering form questions (You do not have access to the form itself, so you must infer. If the message appears to come from a user filling out a form and asking about something within it, classify as 'yes'.); definitions of legal/immigration terms; document requirements; site navigation or technical issues; or general greetings/openers while using the service. Do NOT classify greetings as 'no'.

Mark as 'no' only if the message is clearly unrelated to refugee support or the wcr.is forms.
"""

RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RELEVANCE_SYSTEM_MESSAGE),
    ("human",  "The user's message:\n{query}")
])







    

def get_cfg(config: RunnableConfig) -> Dict[str, Any]:
    """
    Extract the `configurable` dictionary from a RunnableConfig.

    If no config is provided, returns an empty dictionary.

    Args:
        config: The runtime configuration object.

    Returns:
        A dictionary of configurable values (like user language or form string).
    """
    return (config or {}).get("configurable", {})  # type: ignore




#--------------------------------------- NODES ---------------------------------------

def last_human_text(messages: List[BaseMessage]) -> Optional[str]:
    """
    Get the text of the most recent human message.

    Looks through the list of messages in reverse order (latest first).
    Returns the text content of the first HumanMessage found.
    If no human message is present, returns None.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def trim_messages(messages: List[BaseMessage], max_messages: int = 8) -> List[BaseMessage]:
    """
    Keep only the most recent messages.

    Cuts the message history down to the last `max_messages` items.
    This helps keep the context small and efficient when sending to the model.
    """
    return messages[-max_messages:]


async def query_relevance(state: Ask_Ai_AgentState, config: RunnableConfig) -> Ask_Ai_AgentState:
    """
    Prepare the user's last question for the relevance check.

    Steps:
    1. Take the conversation history from state.
    2. Trim it to the most recent messages (default 8).
    3. Extract the latest human message text.
    4. Save that text into state as `user_question`.

    Returns a partial Ask_Ai_AgentState with only the new `user_question` set.
    """
    msgs = trim_messages(state.messages)
    text = last_human_text(msgs) or ""
    return {"user_question": text}




async def query_relevance_router(state: Ask_Ai_AgentState, config: RunnableConfig) -> Literal["cant_help", "answer_user_query"]:
    """
    Decide whether the assistant should answer the user or decline.

    Process:
    1. Take the latest user question from state.
    2. Send it to an LLM with the relevance grading prompt.
    3. The model returns a binary score: "yes" (relevant) or "no" (not relevant).
    4. If "yes", route to the "answer_user_query" node.
       If "no", route to the "cant_help" node.

    Returns:
        A string literal: either "answer_user_query" or "cant_help".
    """
    user_question = state.user_question or ""
    if VERBOSE:
        print("---CHECK RELEVANCE---")

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash-lite",
    #     api_key=GEMINI_API_KEY,
    #     temperature=0,
    #     max_output_tokens=100,
    # )


    llm    = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    # print(config["configurable"].get("model", settings.DEFAULT_MODEL))

    grader = (RELEVANCE_GRADER_PROMPT | llm.with_structured_output(GradeRelevance)).with_config(tags=["skip_stream"])
    relevance_grade: GradeRelevance = await grader.ainvoke({"query": user_question})

    score = (relevance_grade.binary_score or "").strip().lower()
    return "answer_user_query" if score == "yes" else "cant_help"






        
async def cant_help(state: Ask_Ai_AgentState, config: RunnableConfig, writer: StreamWriter) -> Ask_Ai_AgentState:  # writer auto-injected
    """
    Send a polite message when the assistant cannot help.

    Behavior:
    - Checks the user's chosen language from config (default: English).
    - If the language is Russian or Ukrainian, replaces the default text
      with a localized version.
    - Streams the response word by word using the writer (so the user
      sees it appear gradually).
    - Returns the full AIMessage with the chosen "can't help" text.

    Args:
        state:  The current agent state.
        config: Runtime configuration (contains user settings).
        writer: Streaming callback, auto-injected by LangGraph.

    Returns:
        Ask_Ai_AgentState update with the assistant's "can't help" message.
    """
    cfg = get_cfg(config)
    user_language = cfg.get("user_language", "english")

    # print(user_language)

    if user_language.lower() == 'russian':
        state.cant_help_text = 'Извините, я не могу вам помочь в этом вопросе.'
    elif user_language.lower() == 'ukrainian':
        state.cant_help_text = 'Вибачте, я не можу вам допомогти в цьому питанні.'
    for word in state.cant_help_text.split():
        writer(word + ' ')
    return {"messages": [AIMessage(content=state.cant_help_text)]}







MODEL_SYSTEM_MESSAGE = """
Your name is `ask ai`. You are a kind, patient assistant for refugees using the wcr.is website.

The site has multiple forms.
The user chooses the form that matches their current need.
They fill out the form in another tab, not in this chat.
While completing it, they may ask you about words, legal terms, or any confusing part.
Your job is to guide them so they can finish the form correctly.

How to respond:
- The user has selected <user_language> {user_language} </user_language> language. So your response should be in <user_language> {user_language} </user_language> language.
- Be very polite and supportive.
- Use short sentences.
- Use simple, everyday language.
- Explain step by step.
- Focus on the exact question the user is stuck on.
- Give short examples when helpful.
- If you need details, ask one clear question at a time.
- Adjust your tone and explanation style to fit the person you’re talking to.
    - If the person is not well-educated, avoid technical terms. Use simple words and short sentences.
    - If the person seems to be in trauma, respond with care, love, and support. Focus on uplifting their spirit.
    - Infer the person’s eloquence based on how their question is written. 
        - If the question is unclear or sloppy: use simpler language, slow down, and give more (and more concrete) examples.
        - If the question is eloquent and precise: be succinct and get straight to the point, with minimal examples.

- Name of the form is given below. Always tell the user that you are currently helping with this form.

Current form:
<form>
{form_str}
</form>
"""





async def answer_user_query(state: Ask_Ai_AgentState, config: RunnableConfig) -> Ask_Ai_AgentState:
    """
    Generate a helpful answer to the user's question.

    Behavior:
    - Reads the user language and current form name from config.
    - Builds a system message with instructions for tone, style,
      and context (including the form being filled).
    - Combines that with the recent conversation history.
    - Sends everything to an LLM to create a reply.
    - Returns the reply wrapped as an AIMessage.

    Args:
        state:  The current agent state (holds messages and context).
        config: Runtime configuration with user-specific settings.

    Returns:
        Ask_Ai_AgentState update with the assistant's generated answer.
    """
    cfg = get_cfg(config)
    user_language = cfg.get("user_language", "english")
    form_str = cfg.get("form_str", "—")

    # model = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash-lite",
    #     api_key=GEMINI_API_KEY,
    #     temperature=0,
    #     max_output_tokens=1000,  # <-- correct param name
    # )

    model      = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        form_str=form_str,
        user_language=user_language,
    )
    # print(user_language)
    # print(form_str)

    messages = [SystemMessage(content=system_msg)] + trim_messages(state.messages)
    response = await model.ainvoke(messages, config=config)
    return {"messages": [response]}




# -------------------------
# BUILD THE GRAPH
# -------------------------
builder = StateGraph(Ask_Ai_AgentState)
builder.add_node("query_relevance", query_relevance)
builder.add_node("answer_user_query", answer_user_query)
builder.add_node("cant_help", cant_help)

builder.set_entry_point("query_relevance")
builder.add_edge("cant_help", END)
builder.add_edge("answer_user_query", END)
builder.add_conditional_edges(source="query_relevance", path=query_relevance_router, path_map=["cant_help", "answer_user_query"])

# # Compile with an in-memory checkpointer; resume by calling invoke() on the same thread_id
# checkpointer = MemorySaver()

# Compile the graph with persistent checkpointer and in-memory store
ask_ai_agent = builder.compile()#checkpointer=checkpointer)# store=across_thread_memory)

