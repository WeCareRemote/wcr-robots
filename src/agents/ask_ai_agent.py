from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, convert_to_messages, BaseMessage, FunctionMessage, ChatMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Literal, Annotated, List, Dict, Any, Iterable, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter
from langfuse import Langfuse, get_client
import json
from langgraph.graph.message import add_messages

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




#--------------------------------------- Configuring LangFuse ---------------------------------------
if settings.LANGFUSE_TRACING==True:
    langfuse_public_key = settings.LANGFUSE_PUBLIC_KEY #os.environ["langfuse_public_key"]
    langfuse_secret_key = settings.LANGFUSE_SECRET_KEY #os.environ["langfuse_secret_key"]
    langfuse_host       = settings.LANGFUSE_HOST       #os.environ["langfuse_host"]
    
    langfuse = Langfuse(
      secret_key = langfuse_secret_key.get_secret_value(),
      public_key = langfuse_public_key.get_secret_value(),
      host       = langfuse_host
    )
    
    langfuse_cl = get_client(public_key=langfuse_public_key.get_secret_value())





#--------------------------------------- SCHEMAS ---------------------------------------

class Ask_Ai_AgentState(BaseModel):
    """
    Main state for the LangGraph Agent.

    Purpose:
    - Holds the evolving state of the refugee-support assistant while guiding users through forms on wcr.is.
    - Acts as the single source of truth passed between graph nodes.
    - Stores conversation history, user context, and control flags used in routing decisions.

    Attributes:
    - messages: List of conversation messages (Human, AI, System). Maintains dialogue history.
    - context_form: String of the current form the user is filling, or None if no form is active.
    - user_question: The most recent human message text, extracted for relevance checks and answering.
    - cant_help_text: Predefined fallback text sent when the assistant cannot provide help.
    """
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    
    context_form: Optional[str] = Field(
        default     = None,
        title       = "Context Form",
        description = (
            "Full STRING of the current form being filled by the user. "
            "Set to None when no form is active."
        ),
    )
    
    user_question:  Optional[str] = Field(
        default     = None,
        title       = "User Question",
        description = ("The crrent question that the user has asked."),
    )
    
    cant_help_text: str = Field(
        default     = "Sorry, I cannot help you in this matter.",
        title       = "Cannot Help Text",
        description = ("A Predefined Text. If the user's question is not related to refugee help. This predefined text is streamed"),
    )


class GradeRelevance(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Is the user's message relevant to Refugee_Bridge support for using the wcr.is website or completing its forms (including greetings while using the service)? 'yes' or 'no'."
        )
    )
    

#--------------------------------------- FUNCTIONS ---------------------------------------

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


def _coerce_text(content) -> str:
    """Safely turn message.content into text."""
    if isinstance(content, str):
        return content.strip()
    try:
        # Pretty-print dicts/lists; fallback to str for anything else
        return json.dumps(content, ensure_ascii=False, indent=None)
    except Exception:
        return str(content).strip()


def _role_of(msg: BaseMessage) -> str:
    """Map a message object to a simple role label."""
    # LangChain messages expose .type (e.g., 'human', 'ai', 'system', ...)
    t = getattr(msg, "type", None)
    if t:
        return t
    # Fallbacks for custom ChatMessage etc.
    if isinstance(msg, ChatMessage) and hasattr(msg, "role"):
        return msg.role
    # Last resort: class name
    return msg.__class__.__name__.lower()


def chat_history_to_text(
    messages: Iterable[BaseMessage],
    *,
    max_chars_per_message: int | None = None,
    include_ids: bool = False,
) -> str:
    """
    Convert a list of LangChain BaseMessage objects into a simple 'role: text' transcript.

    Params
    - messages: list/iterable of BaseMessage
    - max_chars_per_message: truncate each message's text to this many characters (None = no truncation)
    - include_ids: append message id in square brackets after the role (if present)

    Returns
    - A single string like:
        human: Hi
        ai: Hello!
        system: ...
    """
    lines = []
    for m in messages:
        role = _role_of(m)
        # Normalize a couple of common labels
        role = {"human": "human", "ai": "ai", "system": "system", "tool": "tool", "function": "function"}.get(role, role)

        text = _coerce_text(getattr(m, "content", ""))
        if max_chars_per_message is not None and len(text) > max_chars_per_message:
            text = text[: max_chars_per_message - 1] + "…"

        suffix = f" [{getattr(m, 'id', '')}]" if include_ids and getattr(m, "id", None) else ""
        # Collapse Windows newlines to \n but otherwise leave content as-is
        text = text.replace("\r\n", "\n").strip()

        lines.append(f"{role}:{suffix} {text}")
    return "\n".join(lines)





#--------------------------------------- NODES ---------------------------------------
# # Local Prompt creation
# RELEVANCE_GRADER_SYSTEM_MESSAGE = """
# You are `ask ai`, an assistant for refugees using the wcr.is website. The site has multiple forms; users pick the one matching their current need and fill it out in another tab. While completing a form, they may ask about words, legal terms, or any confusing part. Your job is to guide them so they can finish the form correctly.

# TASK: Decide if the user's message is relevant to Refugee_Bridge assistance. Return ONLY a binary score: 'yes' or 'no'.

# Mark as 'yes' if the message concerns: using wcr.is; choosing/finding the right form; understanding or answering form questions (You do not have access to the form itself, so you must infer. If the message appears to come from a user filling out a form and asking about something within it, classify as 'yes'.); definitions of legal/immigration terms; document requirements; site navigation or technical issues; or general greetings/openers while using the service. Do NOT classify greetings as 'no'.

# Mark as 'no' only if the message is clearly unrelated to refugee support or the wcr.is forms.
# """

# RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", RELEVANCE_GRADER_SYSTEM_MESSAGE),
#     ("human",  "The user's message:\n{query}")
# ])


# Pulling Prompt from LangFuse
try:
    langfuse_RELEVANCE_GRADER_SYSTEM_MESSAGE = langfuse_cl.get_prompt(
        name="RELEVANCE_GRADER_SYSTEM_MESSAGE",
        type="chat"
    )
    
    RELEVANCE_GRADER_PROMPT = ChatPromptTemplate(
        langfuse_RELEVANCE_GRADER_SYSTEM_MESSAGE.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_RELEVANCE_GRADER_SYSTEM_MESSAGE}  # exactly like that for linked generation
    )
except Exception as e:
    print("Unable to pull prompts from Langfuse server: RELEVANCE_GRADER_SYSTEM_MESSAGE")
    print(e)

# query_relevance node.
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
    relevance_grade: GradeRelevance = await grader.ainvoke(
        {
            "query":         user_question,
            "chat_history":  chat_history_to_text(trim_messages(state.messages)[:-1]), 
        }
    )

    score = (relevance_grade.binary_score or "").strip().lower()
    return "answer_user_query" if score == "yes" else "cant_help"




# cant_help node
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






# # Local Prompt creation
# ANSWER_USER_QUERY_SYSTEM_MESSAGE = """
# Your name is `ask ai`. You are a kind, patient assistant for refugees using the wcr.is website.

# The site has multiple forms.
# The user chooses the form that matches their current need.
# They fill out the form in another tab, not in this chat.
# While completing it, they may ask you about words, legal terms, or any confusing part.
# Your job is to guide them so they can finish the form correctly.

# How to respond:
# - The user has selected <user_language> {user_language} </user_language> language. So your response should be in <user_language> {user_language} </user_language> language.
# - Be very polite and supportive.
# - Use short sentences.
# - Use simple, everyday language.
# - Explain step by step.
# - Focus on the exact question the user is stuck on.
# - Give short examples when helpful.
# - If you need details, ask one clear question at a time.
# - Adjust your tone and explanation style to fit the person you’re talking to.
#     - If the person is not well-educated, avoid technical terms. Use simple words and short sentences.
#     - If the person seems to be in trauma, respond with care, love, and support. Focus on uplifting their spirit.
#     - Infer the person’s eloquence based on how their question is written. 
#         - If the question is unclear or sloppy: use simpler language, slow down, and give more (and more concrete) examples.
#         - If the question is eloquent and precise: be succinct and get straight to the point, with minimal examples.

# - Name of the form is given below. Always tell the user that you are currently helping with this form.

# Current form:
# <form>
# {form_str}
# </form>
# """

# Pulling prompt from langfuse
try:
    langfuse_ANSWER_USER_QUERY_SYSTEM_MESSAGE = langfuse_cl.get_prompt(
        name="ANSWER_USER_QUERY_SYSTEM_MESSAGE",
        type="chat",
    )
    
    ANSWER_USER_QUERY_SYSTEM_MESSAGE = ChatPromptTemplate(
        langfuse_ANSWER_USER_QUERY_SYSTEM_MESSAGE.get_langchain_prompt(),
        metadata={"langfuse_prompt": langfuse_ANSWER_USER_QUERY_SYSTEM_MESSAGE}  # exactly like that for linked generation
    )
except Exception as e:
    print("Unable to pull prompts from Langfuse server: ANSWER_USER_QUERY_SYSTEM_MESSAGE")
    print(e)


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
    # system_msg = ANSWER_USER_QUERY_SYSTEM_MESSAGE.format(
    #     form_str=form_str,
    #     user_language=user_language,
    # )
    # print(user_language)
    # print(form_str)
    chain = ANSWER_USER_QUERY_SYSTEM_MESSAGE | model
    response = await chain.ainvoke(
        {
            "user_language": user_language,
            "form_str":      form_str,
            "chat_history":  chat_history_to_text(trim_messages(state.messages)),
        }
    )
    # print(chat_history_to_text(trim_messages(state.messages)))

    # messages = [SystemMessage(content=system_msg)] + trim_messages(state.messages)
    # response = await model.ainvoke(messages, config=config)
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

