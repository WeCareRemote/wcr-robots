import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import ask_ai_agent


@pytest.mark.parametrize(
    "query",
    [
        "Can you explain Bürgergeld for a client from Ukraine?",
        "Which documents are needed for Anmeldung?",
        "Draft a short client-facing reply about a language course.",
        "Hi",
    ],
)
def test_support_topic_heuristic_accepts_wcr_scope(query):
    assert ask_ai_agent.is_obviously_supported_query(query)


def test_support_topic_heuristic_rejects_unrelated_query():
    assert not ask_ai_agent.is_obviously_supported_query("What is the capital of Japan?")


@pytest.mark.asyncio
async def test_relevance_router_answers_obviously_supported_query_without_model(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Supported WCR topics should not depend on the LLM relevance grader")

    monkeypatch.setattr(ask_ai_agent, "get_model", fail_if_called)

    state = ask_ai_agent.Ask_Ai_AgentState(
        messages=[HumanMessage(content="Can a Ukrainian client ask Jobcenter about Bürgergeld?")],
        user_question="Can a Ukrainian client ask Jobcenter about Bürgergeld?",
    )

    route = await ask_ai_agent.query_relevance_router(state, {"configurable": {}})

    assert route == "answer_user_query"


@pytest.mark.asyncio
async def test_cant_help_uses_scoped_fallback_instead_of_hard_refusal():
    streamed_words = []

    def writer(chunk):
        streamed_words.append(chunk)

    state = ask_ai_agent.Ask_Ai_AgentState(messages=[])
    result = await ask_ai_agent.cant_help(state, {"configurable": {"user_language": "english"}}, writer)

    response = result["messages"][0].content
    assert "I can help with general refugee-support" in response
    assert "I cannot decide legal eligibility" in response
    assert response != "Sorry, I cannot help you in this matter."
    assert "".join(streamed_words).strip() == response


@pytest.mark.asyncio
async def test_answer_user_query_adds_trail_semantics_as_system_context(monkeypatch):
    captured = {}

    class FakeModel:
        async def ainvoke(self, messages, config=None):
            captured["messages"] = messages
            return AIMessage(content="answer")

    monkeypatch.setattr(ask_ai_agent, "get_model", lambda *_args, **_kwargs: FakeModel())
    state = ask_ai_agent.Ask_Ai_AgentState(messages=[HumanMessage(content="Compare them")])
    config = {
        "configurable": {
            "trail_semantic_instructions": (
                '<trail_semantic_instructions node_id="planning">\n'
                "- Accessibility (+2.5): Favor the accessible option.\n"
                "</trail_semantic_instructions>"
            )
        }
    }

    await ask_ai_agent.answer_user_query(state, config)

    system_messages = [message for message in captured["messages"] if isinstance(message, SystemMessage)]
    assert any("Accessibility (+2.5)" in str(message.content) for message in system_messages)
