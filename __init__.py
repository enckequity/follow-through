"""follow-through: an agent never ends a turn on a promise.

When the model's final reply (no tool calls) promises work that nothing is running ("I've started the deep dive",
"I'll send you the report shortly"), the reply is not allowed to end the turn. An ``llm_execution`` middleware turns
that final response into a call to this plugin's own ``follow_through`` tool; the tool result tells the agent what
it promised and the only two legal exits: do the work now, in this turn, or actually launch it (background
process, cron, delegated task) and say exactly what is running. The agent loop therefore really continues; a prompt
cannot do this, because a prompt cannot stop a reply that was already written.

Zero model calls. Fail-open on any error. At most FOLLOW_THROUGH_MAX_NUDGES (default 2) per user turn.
Offers and questions ("If you want, I'll ...", "Should I ...?") and honest status ("Done: ...") are left alone.
Kill switch: FOLLOW_THROUGH_OFF=1.
"""
from __future__ import annotations

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
    r"|\bI(?:'ll| will) (?:now |next |then |also |still |keep |go on to )?(?:start|dig|research|study|build|work|"
    r"continue|keep|go|prepare|make|create|write|run|test|send|report|come back|get back|deliver|bring|return|put|"
    r"look|check|review|analy[sz]e|finish|complete|do|turn|compile|draft|record|map|learn|practi[cs]e|master|"
    r"open|hand|pass|launch|kick off|spin up|set up|fix|land|ship|push|update|patch|wire|follow up)\b"
    r"|\bI(?:'ll| will)\b"
    r"|\b(?:running now|started|starting now|already started)\b"
    r"|\bI(?:'m| am) going to\b"
    r"|\b(?:next|then|after that),? I(?:'ll| will)\b"
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
    r"whenever you|unless you|until you)\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:I won'?t|I will not|I'?m not going to|I am not going to|non (?:lo |la )?(?:farò|faccio))\b",
                      re.IGNORECASE)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r";\s+|\s+[\u2014\u2013]\s+|\s*\u2014\s*")

NUDGE = (
    TAG + " Your draft tells the user that work has started or will happen: \"{quote}\". Nothing runs after you "
    "reply: when you answer, your work stops. Before you reply, do one of these now: (1) do that work in this turn "
    "and include its result; or (2) if it needs longer than this turn, actually launch it now (a background "
    "process, a scheduled job, a delegated task) so it keeps running after you reply. Then reply with what is done, "
    "or with exactly what is now running and what it will bring. Never describe work as started unless it is running."
)

SCHEMA = {
    "name": TOOL,
    "description": ("Called automatically by the follow-through plugin when a draft reply promises work that is not "
                    "running. Returns what to do before replying. You do not need to call it yourself."),
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
        return max(0, int(os.environ.get("FOLLOW_THROUGH_MAX_NUDGES", "2")))
    except ValueError:
        return 2


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _nudges_this_turn(messages: Any) -> int:
    """Nudges already given since the last user message (stateless: read from the request itself)."""
    count = 0
    for msg in reversed(list(messages or [])):
        role = _get(msg, "role")
        if role == "user":
            break
        content = _get(msg, "content")
        if role == "tool" and isinstance(content, str) and TAG in content:
            count += 1
    return count


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
        choice = response.choices[0]
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
        logger.info("follow-through: held a final reply that promises %r", quote[:120])
    except Exception as exc:  # fail open: the original reply goes through untouched
        logger.warning("follow-through: skipped (%s)", exc)
    return response


def handle(args: dict, **_: Any) -> str:
    quote = str((args or {}).get("quote") or "the work you described").strip()
    return NUDGE.format(quote=quote[:300])


def register(ctx) -> None:
    ctx.register_tool(name=TOOL, toolset="follow_through", schema=SCHEMA, handler=handle,
                      description=SCHEMA["description"], emoji="\u23f3")
    ctx.register_middleware("llm_execution", middleware)
