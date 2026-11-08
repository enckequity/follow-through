import importlib.util
import json
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("follow_through_plugin", os.path.join(HERE, "__init__.py"))
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

TOOLS = [{"type": "function", "function": {"name": "follow_through"}}]


def resp(text, tool_calls=None):
    msg = SimpleNamespace(role="assistant", content=text, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def run(text, messages=None, tools=TOOLS):
    req = {"messages": messages or [{"role": "user", "content": "go"}], "tools": tools}
    return ft.middleware(request=req, next_call=lambda r: resp(text))


def test_promise_sentences():
    assert ft.promise_sentence("I've started the deep dive and will send you the report shortly.")
    assert ft.promise_sentence("Sure. I'll get back to you with the pricing table.")
    assert ft.promise_sentence("Ho già avviato la ricerca.")


def test_offers_status_and_negations_pass():
    for text in ("Done: the report is at /tmp/r.md.", "If you want, I'll start the deep dive.",
                 "Should I run the full benchmark?", "I won't touch the production database.",
                 "The table is below.", "Here are the three options you asked for."):
        assert ft.promise_sentence(text) is None, text


def test_promise_becomes_tool_call():
    r = run("I've started the deep dive and will send you the full report once it's done.")
    calls = r.choices[0].message.tool_calls
    assert r.choices[0].finish_reason == "tool_calls"
    assert calls[0].function.name == "follow_through"
    assert "deep dive" in json.loads(calls[0].function.arguments)["quote"]


def test_honest_reply_untouched():
    r = run("Done: the report is at /tmp/r.md.")
    assert r.choices[0].message.tool_calls is None and r.choices[0].finish_reason == "stop"


def test_cap_per_turn():
    nudge = ft.handle({"quote": "x"})
    msgs = [{"role": "user", "content": "go"},
            {"role": "tool", "content": nudge}, {"role": "tool", "content": nudge}]
    assert run("I'll send it later today.", messages=msgs).choices[0].message.tool_calls is None
    # a new user message resets the count
    msgs.append({"role": "user", "content": "again"})
    assert run("I'll send it later today.", messages=msgs).choices[0].message.tool_calls


def test_needs_tool_offered_and_kill_switch(monkeypatch):
    assert run("I'll send it later today.", tools=[]).choices[0].message.tool_calls is None
    monkeypatch.setenv("FOLLOW_THROUGH_OFF", "1")
    assert run("I'll send it later today.").choices[0].message.tool_calls is None


def test_fail_open_on_odd_response():
    req = {"messages": [], "tools": TOOLS}
    odd = object()
    assert ft.middleware(request=req, next_call=lambda r: odd) is odd


def test_nudge_text():
    out = ft.handle({"quote": "I'll send you the report shortly."})
    assert out.startswith("[follow-through]") and "send you the report" in out and "launch" in out


def test_deferred_tool_goes_through_tool_search_bridge():
    r = run("I'll send you the report shortly.", tools=[{"type": "function", "function": {"name": "tool_call"}}])
    call = r.choices[0].message.tool_calls[0]
    assert call.function.name == "tool_call"
    assert json.loads(call.function.arguments)["calls"][0]["name"] == "follow_through"
