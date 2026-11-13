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


def test_future_availability_is_not_work():
    for text in ("I'll be available tomorrow.", "I will be here if needed.",
                 "I'm going to be available tomorrow.", "I am going to be offline tomorrow.",
                 "Next, I'll be here.", "Then I will be available.", "I'll go offline tomorrow."):
        assert ft.promise_sentence(text) is None, text


def test_concrete_work_promises_still_match():
    for text in ("I'll review the report now.", "I will prepare the comparison.",
                 "I'm going to check the results.", "I am going to draft the reply.",
                 "Next, I'll build the table.", "I’ve started the research.",
                 "Sto preparando il rapporto."):
        assert ft.promise_sentence(text), text


def test_approval_handoffs_are_conditional():
    for text in ("I'll send the report after you approve.", "After you approve, I'll send the report.",
                 "I'll send it after your approval.", "After approval, I'll send the report.",
                 "I'll send it once approved.", "If approved, I'll send the report.",
                 "I'll send it pending your approval.", "I'll send it with your approval.",
                 "Once you approve, I'll send the report.", "If you approve, I'll send the report."):
        assert ft.promise_sentence(text) is None, text


def test_limit_defaults_and_hard_cap_one(monkeypatch):
    monkeypatch.delenv("FOLLOW_THROUGH_MAX_NUDGES", raising=False)
    assert ft._max_nudges() == 1
    for value, expected in (("0", 0), ("-1", 0), ("1", 1), ("2", 1), ("999", 1), ("", 1), ("invalid", 1)):
        monkeypatch.setenv("FOLLOW_THROUGH_MAX_NUDGES", value)
        assert ft._max_nudges() == expected
    monkeypatch.setenv("FOLLOW_THROUGH_MAX_NUDGES", "0")
    assert run("I'll send the report now.").choices[0].message.tool_calls is None


def test_issued_nudge_counts_without_a_successful_result(monkeypatch):
    monkeypatch.setenv("FOLLOW_THROUGH_MAX_NUDGES", "1")
    for name, arguments in ((ft.TOOL, {"quote": "Synthetic promise"}),
                            (ft.BRIDGE, {"calls": [{"name": ft.TOOL, "arguments": {"quote": "Synthetic promise"}}]})):
        call = {"id": "synthetic-nudge", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for tool_result in ([], [{"role": "tool", "tool_call_id": call["id"], "content": "Tool unavailable"}]):
            messages = [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [call]}, *tool_result]
            assert ft._nudges_this_turn(messages) == 1
            assert run("I'll send the report now.", messages=messages).choices[0].message.tool_calls is None
            messages.append({"role": "user", "content": "again"})
            assert run("I'll send the report now.", messages=messages).choices[0].message.tool_calls


def test_call_and_result_are_not_counted_twice():
    call = SimpleNamespace(function=SimpleNamespace(name=ft.TOOL, arguments='{}'))
    messages = [{"role": "user", "content": "go"}, SimpleNamespace(role="assistant", tool_calls=[call]),
                {"role": "tool", "content": ft.handle({"quote": "Synthetic promise"})}]
    assert ft._nudges_this_turn(messages) == 1


def test_rewrite_does_not_mutate_downstream_response():
    original = resp("I'll prepare the report now.")
    result = ft.middleware({"tools": TOOLS}, lambda request: original)
    assert result is not original and result.choices[0].message.tool_calls
    assert original.choices[0].message.tool_calls is None and original.choices[0].finish_reason == "stop"


def test_immutable_choice_fails_open_without_partial_injection():
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Choice:
        message: object
        finish_reason: str = "stop"
    original = SimpleNamespace(choices=[Choice(SimpleNamespace(content="I'll prepare the report now.", tool_calls=None))])
    result = ft.middleware({"tools": TOOLS}, lambda request: original)
    assert result is original and original.choices[0].message.tool_calls is None


def test_shared_copy_shape_is_left_untouched():
    class SharedMessage(SimpleNamespace):
        def __deepcopy__(self, memo): return self
    original = SimpleNamespace(choices=[SimpleNamespace(message=SharedMessage(content="I'll prepare the report now.", tool_calls=None), finish_reason="stop")])
    result = ft.middleware({"tools": TOOLS}, lambda request: original)
    assert result is original and original.choices[0].message.tool_calls is None


def test_malformed_shapes_and_hook_error_fail_open(monkeypatch):
    for original in (object(), {}, SimpleNamespace(choices=[]), SimpleNamespace(choices=[None]),
                     SimpleNamespace(choices=[SimpleNamespace(message=None)])):
        assert ft.middleware({"tools": TOOLS}, lambda request: original) is original
    original = resp("I'll prepare the report now.")
    def broken(*args): raise ValueError("Synthetic customer text must not be logged")
    monkeypatch.setattr(ft, "check", broken)
    assert ft.middleware({"tools": TOOLS}, lambda request: original) is original
    assert original.choices[0].message.tool_calls is None


def test_next_call_once_and_downstream_errors_propagate():
    import pytest
    calls = []
    failure = RuntimeError("Synthetic downstream failure")
    def downstream(request):
        calls.append(request)
        raise failure
    with pytest.raises(RuntimeError) as error:
        ft.middleware({"tools": TOOLS}, downstream)
    assert error.value is failure and len(calls) == 1
    calls.clear()
    assert ft.middleware({"tools": TOOLS}, lambda request: calls.append(request) or resp("Done: the report is ready."))
    assert len(calls) == 1


def test_nudge_respects_running_evidence_blockers_and_approvals():
    text = ft.handle({"quote": "Synthetic promise"}).lower()
    assert "already running" in text and "evidence" in text and "blocker" in text
    assert "approval" in text and "consent" in text
    assert "nothing runs after" not in text and "only two legal exits" not in text


def test_logs_do_not_contain_reply_or_exception_content(caplog, monkeypatch):
    import logging
    caplog.set_level(logging.INFO, logger=ft.logger.name)
    private = "Private customer detail: I'll prepare the report now."
    run(private)
    assert "Private customer detail" not in caplog.text and "prepare the report" not in caplog.text
    def broken(*args): raise ValueError("Private exception detail")
    monkeypatch.setattr(ft, "check", broken)
    run(private)
    assert "Private exception detail" not in caplog.text


def test_other_tool_calls_and_malformed_bridge_history():
    original = resp("I'll prepare the report now.", tool_calls=[SimpleNamespace(function=SimpleNamespace(name="other"))])
    assert ft.middleware({"tools": TOOLS}, lambda request: original) is original
    for arguments in ('{', 'null', '[]', '{"calls": null}', '{"calls": [{"name": "other"}]}'):
        messages = [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [
            {"function": {"name": ft.BRIDGE, "arguments": arguments}}]}]
        assert ft._nudges_this_turn(messages) == 0
        assert run("I'll prepare the report now.", messages=messages).choices[0].message.tool_calls


def test_only_native_tool_and_middleware_without_file_network_or_process_access(monkeypatch):
    import builtins
    import io
    import socket
    import subprocess
    registered = []
    class Context:
        def register_tool(self, **kwargs): registered.append(("tool", kwargs))
        def register_middleware(self, *args): registered.append(("middleware", args))
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Unexpected file, network or process access")
    for target, name in ((builtins, "open"), (io, "open"), (socket, "socket"),
                         (subprocess, "Popen"), (os, "system")):
        monkeypatch.setattr(target, name, denied)
    ft.register(Context())
    result = run("I'll prepare the report now.")
    assert result.choices[0].message.tool_calls
    assert ft.handle({"quote": "Synthetic promise"}).startswith(ft.TAG)
    assert [kind for kind, value in registered] == ["tool", "middleware"]
    assert registered[0][1]["name"] == ft.TOOL
    assert registered[1][1] == ("llm_execution", ft.middleware)
    assert attempts == []
