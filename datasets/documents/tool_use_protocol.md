# A minimal tool-calling protocol for a small model

## Why a textual protocol

Large models are often trained with a structured chat template that the serving
stack parses. A small local model can use a simpler approach: tool calls are
ordinary text delimited by special tokens. The model writes a JSON object
between a `<tool_call>` marker and the end of its turn. The runtime parses the
JSON, executes the named tool, and appends the observation after a
`<tool_result>` marker. Nothing about this requires a vendor API.

The format is:

    <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}

One JSON object per call, `name` a string, `arguments` an object. Multiple
calls may appear on separate lines when they are independent.

## Parsing defensively

Model output is not guaranteed to be valid JSON. A parser should locate the
marker, extract the first balanced JSON object, and report a readable error if
parsing fails. It should reject a call with a missing name, a non-object
argument list, an unknown tool name, missing required arguments, arguments of
the wrong type and unknown argument names. Feeding these errors back as a
`<tool_result>` lets the model correct itself instead of failing silently.

Duplicate calls deserve special handling. A model that receives a result and
then repeats the identical call is stuck; the runtime should return an error
and stop the loop rather than execute the same query forever.

## Tool result shape

A uniform result object keeps the loop simple:

```json
{"ok": true, "content": "17*23+4 = 395", "citations": []}
```

`content` is what the model reads. `ok` is false for expected failures, with an
`error` string. `citations` is a list of structured references, each with a
title, path, chunk ordinal and character offsets, so a final answer can cite
real passages rather than invented ones.

## Budgets and safety

Every loop needs limits: a maximum number of iterations, a maximum number of
tool calls per turn, a per-tool timeout and a cap on context tokens. Tools
should be least-privilege by construction. A calculator parses arithmetic with
an allowlist instead of calling `eval`. File tools resolve paths against a
document root and reject anything that escapes it. Research search is
read-only. There is no shell tool and no arbitrary code execution.
