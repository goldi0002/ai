# BM25 ranking and SQLite FTS5

## What full-text search needs

A full-text index maps terms to the documents that contain them. The basic
structure is an inverted index: for every term, a posting list of document ids,
term frequencies and positions. At query time the engine looks up each query
term, intersects or unions the posting lists, and ranks the candidates.

Ranking matters because a term like "the" appears everywhere while a rare term
is informative. BM25 (Best Matching 25) is the ranking function used by most
search engines and by SQLite's FTS5 extension. For a query term q and document
d it combines three signals:

- term frequency: more occurrences of q in d increase the score, with
  diminishing returns controlled by the parameter k1;
- inverse document frequency: terms that occur in few documents weigh more;
- document length normalisation: long documents do not win merely by being
  long, controlled by the parameter b.

FTS5 exposes the score with the `bm25(table)` auxiliary function. Its return
value is negative, and better matches are more negative, so queries order by
`bm25(...)` ascending and negate it if a positive score is wanted.

## Schema pattern

The recommended pattern is an external content table. The chunk text lives in
an ordinary table, and the FTS5 table indexes it without storing a second copy:

```sql
CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
```

Triggers keep the index in sync on insert, update and delete. With
`content='chunks'`, deleting a row from `chunks` does not remove it from the
index automatically; the delete trigger must issue the special
`INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ...)` command.

## Query construction and safety

FTS5 query syntax includes operators such as AND, OR, NOT, NEAR and prefix
wildcards. Passing raw user text to MATCH allows a stray quote or operator to
raise a syntax error, so production code quotes each term and joins the quoted
terms explicitly. A quoted term is treated as literal text. Phrase queries are
built by quoting the whole phrase.

Stemming is enabled with the `porter` tokenizer, which maps "ranking" and
"ranked" to the same stem. Unicode normalisation is handled by `unicode61`.
These two tokenizers can be combined, for example `tokenize='unicode61 porter'`.

## Chunking

Documents are usually split into overlapping chunks before indexing. A window
of roughly 600 to 1000 characters with 10 to 20 percent overlap keeps retrieval
precise and ensures that a passage crossing a boundary can still be found. Each
chunk keeps its document id, ordinal and character offsets so a search result
can cite an exact location instead of an entire file.
