# follow-through

Nudge concrete work promises through the existing native agent loop.

When a final reply promises work, an `llm_execution` middleware turns it into a call to the plugin's `follow_through` tool. Deferred tools use Hermes' existing `tool_call` bridge. The tool asks the agent to give the result, report evidence of work already running, take an authorized action, or explain a blocker. It grants no authority and does not know whether work is running.

- At most one nudge per user turn, including failed or unanswered tool calls.
- Approval handoffs, offers, questions, negations and ordinary availability statements pass through.
- English and Italian promise patterns; text matching can miss or misclassify a promise.
- `FOLLOW_THROUGH_MAX_NUDGES=0` or `FOLLOW_THROUGH_OFF=1` disables nudges. The default and hard maximum are one; invalid numeric input uses the default.
- Malformed or unsupported responses pass through. A failed rewrite leaves the original response intact; downstream request errors still propagate.

## Permissions and execution

The nudge preserves approval and consent requirements for sending, spending, scheduling and delegation. It asks for existing execution evidence rather than launching duplicate work. Routines remain offers until explicitly accepted. Native tool permissions and approval gates still apply.

The plugin makes no independent model request. Rewriting a reply into a tool call can continue the existing agent loop, including its normal model requests. The hook and tool perform no file, network or subprocess operations and keep no durable state. Generic diagnostic logs contain no draft quotation or exception details. The promise quotation stays in the ordinary tool-call history.

The hook handles mutable chat-completions-style response objects. Unsupported response shapes are left unchanged; this is not a guarantee across every provider or streaming transport.

## Install

Requires Hermes `>=0.21.6`. Use Hermes' native plugin installer with a reviewed, full commit pin:

```bash
hermes plugins install enckequity/follow-through --ref <reviewed-40-character-commit> --no-deps --no-enable
```

Review the native scanner result before enabling the plugin. Installation and enabling are separate steps; this repository does not change any engine, provider, job or schedule.

## Examples

| Draft | Behavior |
| --- | --- |
| “I'll prepare the comparison now.” | One nudge asking for a result, existing execution evidence, an authorized action or a blocker. |
| “I'll send the report after you approve.” | Passes through; approval remains pending. |
| “I'll be available tomorrow.” | Passes through. |
| A second promise after the nudge fails or has no result | Passes through; the issued nudge already reached the turn limit. |

## Tests

```bash
python -m pytest -q tests
```

Tests use synthetic replies and history. They cover direct and bridged nudges, approval language, turn limits, response-copy failures, downstream errors, log privacy and blocked file/network/process access. They do not make live model requests.
