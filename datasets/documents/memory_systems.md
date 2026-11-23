# Persistent memory for a local assistant

## Why memory is a separate store

A language model's weights are frozen after training. Anything learned during a
conversation disappears when the process exits unless it is written somewhere.
Persistent memory is that somewhere: a small database of keyed notes that the
assistant can read at the start of a turn and write when the user asks it to
remember something.

Keeping memory separate from the model has three benefits. The user can inspect
and edit it. Updates are immediate, with no retraining. And a mistaken note can
be deleted instead of silently influencing every future answer.

## A practical schema

A minimal memory table stores a stable key, the note content, optional tags,
an importance score and timestamps:

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

The unique key is the important part. Saving the same key again updates the
note in place, so "user_timezone" cannot accumulate five contradictory values.
An FTS5 index over key, content and tags supports keyword retrieval; ordering
by importance and recency covers the empty-query case.

## Retrieval policy

Retrieval should be conservative. Injecting every note into every prompt wastes
context and increases the chance that an irrelevant fact distracts the model. A
common policy is to search memory with the user's message, take the top two or
three notes, and cap the injected text at a few hundred characters. Notes carry
their updated timestamp so the model can say when a fact was recorded.

## Writing policy

The assistant should write memory only when the user asks it to, or when the
user states a durable preference and confirms it. Facts inferred from a single
ambiguous sentence should not be persisted. Because keys are visible to the
user, they should be short and descriptive: `user_timezone` is better than
`note_17`.

## Memory is not a database of truth

Stored notes can become stale. A timezone changes, a project is renamed, a
preference is reversed. Treat memory as a hint with a provenance timestamp, not
as ground truth, and prefer the user's current message when the two conflict.
This is also why deletion and update must be first-class operations rather than
an afterthought.
