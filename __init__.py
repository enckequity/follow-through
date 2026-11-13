"""Nudge concrete work promises through the native agent loop, at most once per user turn.

The hook checks reply text, not running state. The nudge asks for a result, existing execution
evidence, an authorized next action or an honest blocker. It makes no independent model request;
the existing loop may continue. Errors leave the downstream response untouched.
FOLLOW_THROUGH_MAX_NUDGES=0 or FOLLOW_THROUGH_OFF=1 disables it.
"""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
import re
import uuid
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOOL = "follow_through"
TAG = "[follow-through]"

# Work the agent says has started or will happen (English and Italian).
_PROMISE = re.compile(
    r"\bI(?:'ve| have) (?:now )?(?:started|begun|kicked off|reopened|launched|initiated|set (?:it )?in motion)\b"
    r"|\bI(?:'m| am) (?:now |already )?(?:starting|digging|researching|studying|working on|building|running|"
    r"preparing|putting together|continuing|going deeper|looking into|pulling|gathering|collecting)\b"
    r"|\bI(?:'ll| will|(?:'m| am) going to) (?:now |next |then |also |still |keep |go on to )?(?:start|dig|research|study|build|work|"
    r"continue|keep|prepare|make|create|write|run|test|send|report|come back|get back|deliver|bring|return|put|"
    r"look|check|review|analy[sz]e|finish|complete|do|turn|compile|draft|record|map|learn|practi[cs]e|master|"
    r"open|hand|pass|launch|kick off|spin up|set up|fix|land|ship|push|update|patch|wire|follow up)\b"
    r"|\b(?:running now|started|starting now|already started)\b"
    r"|\b(?:keep|keeps) (?:digging|going|working|climbing|studying|learning)\b"
    r"|\b(?:report|come|get) back (?:to you|when|once|with|as soon)\b"
    r"|\bho (?:già )?(?:iniziato|cominciato|avviato|ripreso|lanciato)\b"
    r"|\bsto (?:lavorando|studiando|cercando|preparando|costruendo|analizzando|scavando|raccogliendo)\b"
    r"|\b(?:inizio|comincio|continuo|procedo|preparo|costruisco|approfondisco|torno|riprendo) (?:ora|adesso|subito|a|con|il|la|lo|i|le|gli)\b"
    r"|\bti (?:aggiorno|scrivo|mando|porto|riporto|faccio sapere|riscrivo)\b",
    re.IGNORECASE)
# A sentence that hands the next step to the user is not a promise of the agent's own work.
_CONDITIONAL = re.compile(
    r"\b(?:if you|if that|se vuoi|se ti va|when you|quando (?:vuoi|mi dici)|tell me|dimmi|your call|should I|"
    r"do you want|want me to|vuoi che|on your (?:go|word|yes)|once you|as soon as you|appena (?:mi|tu)|say go|"
    r"after you|after (?:your )?(?:approval|confirmation)|(?:once|when|if) (?:approved|confirmed)|"
    r"pending (?:your )?(?:approval|confirmation)|with your approval|subject to (?:your )?approval|"
    r"whenever you|unless you|until you)\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:I won'?t|I will not|I'?m not going to|I am not going to|non (?:lo |la )?(?:farò|faccio))\b",
                      re.IGNORECASE)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r";\s+|\s+[\u2014\u2013]\s+|\s*\u2014\s*")

NUDGE = (
    TAG + " Your draft promises work: \"{quote}\". Check what you can substantiate before replying. "
    "If the work is done, give the result. If it is already running, report the execution evidence; do not "
    "launch a duplicate. Otherwise do the work now only if authorized, or explain the blocker and what is needed. "
    "Keep all approval and consent requirements: do not send, spend, schedule or delegate without the required "
    "permission. Offer a routine for explicit acceptance rather than creating one without consent. "
    "This nudge does not know whether work is running and grants no authority."
)

SCHEMA = {
    "name": TOOL,
    "description": ("Called automatically when a draft reply promises concrete work. Returns guidance to report results, "
                    "existing execution evidence or an honest blocker while preserving required permissions. "
                    "You do not need to call it yourself."),
    "parameters": {
        "type": "object",
        "properties": {"quote": {"type": "string", "description": "The promise sentence from the draft."}},
        "required": ["quote"],
    },
}


def promise_sentence(text: str) -> Optional[str]:
    """The first sentence that promises the agent's own not-yet-done work, or None."""
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    for sentence in _SPLIT.split(text):
        s = sentence.strip()
        if len(s) < 12 or s.endswith("?"):
            continue
        if _CONDITIONAL.search(s):
            continue
        for clause in _CLAUSE.split(s):
            if _PROMISE.search(clause) and not _NEGATED.search(clause):
                return s
    return None


def _max_nudges() -> int:
    try:
        return min(1, max(0, int(os.environ.get("FOLLOW_THROUGH_MAX_NUDGES", "1"))))
    except ValueError:
        return 1


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _nudges_this_turn(messages: Any) -> int:
    """Whether one nudge was issued since the last user message, even if its tool failed."""
    for msg in reversed(list(messages or [])):
        role = _get(msg, "role")
        if role == "user":
            break
        calls = _get(msg, "tool_calls") if role == "assistant" else None
        for call in calls if isinstance(calls, (list, tuple)) else []:
            function = _get(call, "function")
            name = _get(function, "name")
            if name == TOOL:
                return 1
            if name == BRIDGE:
                args = _get(function, "arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        continue
                nested = args.get("calls") if isinstance(args, dict) else None
                if isinstance(nested, list) and any(_get(item, "name") == TOOL for item in nested):
                    return 1
        content = _get(msg, "content")
        if role == "tool" and isinstance(content, str) and TAG in content:
            return 1  # Older histories may retain the tool result without its assistant call.
    return 0


BRIDGE = "tool_call"  # Hermes Tool Search: plugin tools may be deferred behind tool_search/describe/call


def _route(request: dict) -> Optional[str]:
    """How this request can reach the tool: directly, through the Tool Search bridge, or not at all."""
    names = set()
    for tool in request.get("tools") or []:
        fn = _get(tool, "function") or {}
        names.add(_get(fn, "name") or _get(tool, "name"))
    return TOOL if TOOL in names else (BRIDGE if BRIDGE in names else None)


def check(request: dict, response: Any) -> Optional[str]:
    """The promise sentence when this final response must not end the turn, else None."""
    if os.environ.get("FOLLOW_THROUGH_OFF") == "1" or not isinstance(request, dict):
        return None
    choices = getattr(response, "choices", None)
    if not choices:
        return None  # not a chat-completions response: fail open
    message = getattr(choices[0], "message", None)
    if message is None or getattr(message, "tool_calls", None):
        return None  # the agent is still acting
    if not _route(request) or _nudges_this_turn(request.get("messages")) >= _max_nudges():
        return None
    content = getattr(message, "content", None)
    return promise_sentence(content) if isinstance(content, str) else None


def middleware(request: dict, next_call, **_: Any) -> Any:
    response = next_call(request)
    try:
        quote = check(request, response)
        if not quote:
            return response
        held = deepcopy(response)
        choice = held.choices[0]
        if held is response or choice is response.choices[0] or choice.message is response.choices[0].message:
            return response
        args = {"quote": quote[:300]}
        if _route(request) == BRIDGE:  # deferred tool: invoke it through the public Tool Search bridge
            name, args = BRIDGE, {"calls": [{"name": TOOL, "arguments": args}]}
        else:
            name = TOOL
        call = SimpleNamespace(
            id="call_ft_" + uuid.uuid4().hex[:12], type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        choice.message.tool_calls = [call]
        choice.finish_reason = "tool_calls"
        logger.info("follow-through: nudged a concrete work promise")
        return held
    except Exception:  # fail open: the original reply goes through untouched
        logger.warning("follow-through: skipped a reply check")
    return response


def handle(args: dict, **_: Any) -> str:
    quote = str((args or {}).get("quote") or "the work you described").strip()
    return NUDGE.format(quote=quote[:300])


def register(ctx) -> None:
    ctx.register_tool(name=TOOL, toolset="follow_through", schema=SCHEMA, handler=handle,
                      description=SCHEMA["description"], emoji="\u23f3")
    ctx.register_middleware("llm_execution", middleware)
