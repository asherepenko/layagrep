# layagrep

**Find code by asking what it does — judged by a local Laya typed-decision model.**

`layagrep` is a Rust sibling of [jevgrep](https://github.com/dzhng/jevgrep): you
ask a natural-language question about a repository, and it prints relevant
files, verbatim source excerpts with line references, and declaration leads —
evidence for a coding agent, not a generated answer.

The judge is a local [Laya](https://huggingface.co/convaiinnovations/laya)
encoder (~400M parameters, one bidirectional forward pass per question, 0
output tokens) instead of a cloud LLM. Everything runs on your machine.

```sh
layagrep "How are telemetry events recorded and sent?" ./my-project
```

## Install

```sh
cargo install --path .
```

No API key. Two interchangeable judge backends, auto-selected:

| Backend | What it uses | Speed (per question) | Notes |
|---|---|---|---|
| `python` (preferred) | the installed [`laya-mlx`](https://github.com/mizorewww/laya-mlx) package, driven through a resident worker process | ~10–40 ms (Apple GPU) | reference port; found via the `laya-mlx` launcher's interpreter, `LAYAGREP_PYTHON`, or any `python3` with `laya_mlx` importable |
| `native` | [`laya-rs`](https://crates.io/crates/laya-rs) (candle) compiled into this binary | ~50–100 ms (CPU, Accelerate) | no Python required; single binary |

The `aac6fef/laya-mlx` checkpoint downloads from Hugging Face on first use and
is cached. `layagrep doctor` verifies the whole path end to end.

## How retrieval works

Measured behavior of the judge drove the design (details in
`.artifacts/progress.md`):

1. **Describe** — every eligible file (respecting `.gitignore`/`.ignore`,
   skipping hidden paths, dependency/build directories, sensitive names,
   binaries, non-UTF-8) is parsed with tree-sitter (Python,
   TypeScript/JavaScript) into a short description: path, header, declaration
   names, imports.
2. **Rank** — BM25 over those descriptions ranks files lexically
   (identifier-aware tokenization, inflection variants); scores propagate along
   the import graph so structurally-relevant files score in.
3. **Gate** — the Laya judge grades each ranked file's description. Absolute
   file-level judgment is noisy for a 421M encoder, so it gates (rejecting
   lexically-matching but off-topic files) and modulates the ranking rather
   than ranking alone.
4. **Select** — declaration units (functions, classes with member units, or
   text chunks for other files) are judged by description (name, signature,
   local calls). Selected units merge into excerpt windows expanded with
   adjacent comments.
5. **Report** — ranked file list with roles, verbatim source blocks with line
   references, declaration locations (`name@start-end`), Python local-call
   leads, `AGENTS.md` lookups, and pytest entry points.

All answers are cached locally (`~/.cache/layagrep`, 7-day TTL, 256 MB cap)
keyed by exact state + questions: repeated searches are near-instant.

```sh
layagrep cache clear     # reset the answer cache
layagrep doctor          # verify the judge backend
layagrep skill --dir .   # write the agent SKILL.md
```

## CLI

```
layagrep [OPTIONS] <QUERY> [ROOT]

  --engine auto|native|python   judge backend selection (default: auto)
  --model <ID>                  checkpoint (default: aac6fef/laya-mlx)
  --dtype float16|float32|bfloat16
  --python <PATH>               interpreter for the python backend
  --hidden                      include hidden paths
  --no-ignore                   disable .gitignore/.ignore patterns
  --include-dependencies        include dependency/build directories
  --include-sensitive           include known sensitive filenames
  --no-cache                    disable answer cache
  --max-source-bytes N          source allocation; 0 = unlimited
  --debug-scores                per-file scores + phase timings on stderr
  -q, --quiet                   suppress stderr progress
```

Exit codes: `0` complete, `1` failed, `2` incomplete, `130` interrupted.
All report output goes to stdout; progress and engine logs go to stderr.

## Honest limitations

- The judge is a 421M classifier, not an LLM. Rankings are lexical-first with
  a semantic gate; files whose relevance is purely structural can rank lower
  than a cloud-LLM judge would place them, and topically-named noise (a texture
  cache for "caching" queries) can appear below the true hits.
- First search on a large repository takes tens of seconds (one judge call per
   lexically-matching file, plus per-declaration selection); repeats hit the
  cache in ~2–3 s.
- Declaration parsing covers Python and TypeScript/JavaScript; other text
  files fall back to bounded chunks.

## License

MIT.
