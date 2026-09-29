# layagrep

**Find code by asking what it does — judged by a local Laya typed-decision model.**

`layagrep` is a Rust sibling of [jevgrep](https://github.com/dzhng/jevgrep):
ask a natural-language question about a repository and it prints ranked files,
verbatim source excerpts with line references, and declaration leads —
evidence for a coding agent, not a generated answer. The judge is a local
[Laya](https://huggingface.co/convaiinnovations/laya) encoder (~421M
parameters, one bidirectional forward pass per question, 0 output tokens)
instead of a cloud LLM. No API keys; everything runs on your machine.

```sh
layagrep "How are telemetry events recorded and sent?" ./my-project
```

## Requirements

- Rust 1.85+ with a C compiler (cmake not required; tree-sitter grammars
  compile via `cc`)
- **python backend** (default when available): macOS on Apple Silicon with the
  [`laya-mlx`](https://github.com/mizorewww/laya-mlx) package installed —
  `pip install laya-mlx` or `uv tool install laya-mlx`
- **native backend** (fallback, no Python needed): macOS (CPU + Accelerate);
  Linux CPU should build the same way but is untested
- ~1 GB disk for the checkpoint on first use (downloaded from Hugging Face
  and cached)

## Install

```sh
git clone <this-repo> layagrep && cd layagrep
cargo install --path .
layagrep doctor   # verify the judge end to end
```

The build vendors the Kotlin and Swift tree-sitter grammars under `vendor/`
(their crates.io releases pin incompatible tree-sitter core versions; the
vendored copies relax that build-time-only range — see `vendor/*/Cargo.toml`).

## Judge backends

Two interchangeable engines, selected by `--engine auto|native|python`
(auto prefers python, falls back to native):

| Backend | What it uses | Speed per question | Notes |
|---|---|---|---|
| `python` | the installed `laya-mlx` package, driven through a resident worker process (JSONL over stdio) | ~10–40 ms (Apple GPU) | the reference port (validated 63/63 parity upstream) |
| `native` | [`laya-rs`](https://crates.io/crates/laya-rs) (candle) compiled into this binary | ~50–100 ms (CPU, Accelerate) | single static binary, no Python; float16/bfloat16 promote to float32 on CPU |

Interpreter discovery for the python backend, in order:
1. `--python PATH` / `LAYAGREP_PYTHON`
2. the interpreter named in the `#!` shebang of the `laya-mlx` launcher on `PATH`
3. any `python3` that can `import laya_mlx`

Both backends answer identically to the 4th decimal (verified against the
reference implementation), and cached answers transfer across engines.

## How retrieval works

The design was driven by measured judge behavior (experiment log in
`.artifacts/progress.md`):

1. **Describe** — eligible files (respecting `.gitignore`/`.ignore` and the
   global git exclude file; skipping hidden paths, dependency/build
   directories, sensitive names, binaries, non-UTF-8, symlinks, files >16 MB)
   are parsed with tree-sitter into a short description: path, header,
   declaration names, imports.
2. **Rank** — BM25 over descriptions (identifier-aware tokenization, inflection
   variants) ranks files lexically; scores propagate along the import graph so
   structurally relevant files (a caller whose "caching" lives in an imported
   module) score in.
3. **Gate** — the Laya judge grades each ranked file's description. Absolute
   file-level judgment is noisy for a 421M encoder, so it *gates* (rejecting
   lexically matching but off-topic files) and modulates the lexical ranking.
   The judge is **lazy**: it only spawns (process + model load) on the first
   cache miss, so fully cached reruns finish in ~0.1–0.3 s with no engine.
4. **Select** — the top candidates' declaration units (functions, classes with
   member units; text chunks for unparsed files) are judged by description
   (name, signature, local calls). At most the 14 most lexically promising
   units per file are judged. Selected units merge into excerpt windows
   expanded with adjacent comments.
5. **Report** — ranked file list with roles, verbatim source blocks with line
   references, declaration locations (`name@start-end`), Python local-call
   leads, `AGENTS.md` lookups, and pytest entry-point suggestions.

## CLI

```
layagrep [OPTIONS] <QUERY> [ROOT]

Search options:
  --hidden                    Include hidden (dot-prefixed) paths
  --no-ignore                 Disable .gitignore/.ignore patterns
  --include-dependencies      Include dependency/build dirs (node_modules, target, ...)
  --include-sensitive         Include sensitive filenames (.env, *.pem, id_rsa, ...)
  --no-cache                  Disable the answer cache for this run
  --max-source-bytes N        Total source-byte budget in the text report (0 = unlimited)

Engine options:
  --engine auto|native|python Judge backend (default: auto)
  --model ID                  Checkpoint: HF id or local path (default: aac6fef/laya-mlx)
  --dtype T                   float16 | float32 | bfloat16 (default float16)
  --python PATH               Interpreter hosting laya_mlx (or LAYAGREP_PYTHON)
  --workers N                 Parallel judge workers (0 = auto: 1 python, 4 native)

Output options:
  --json                      Structured JSON (schema v1) instead of the text report
  --debug-scores              Per-file scores, blends, and phase timings on stderr
  -q, --quiet                 Suppress stderr progress

Commands:
  layagrep doctor [--engine ...] [--model ...] [--dtype ...] [--python ...]
                              Verify the judge with a synthetic question
  layagrep cache clear        Clear cached judge answers
  layagrep skill [--dir DIR]  Print the agent skill; --dir writes SKILL.md

Environment:
  LAYAGREP_PYTHON             Interpreter for the python backend (same as --python)

Exit codes:
  0 complete   1 failed   2 incomplete (issues reported)   130 interrupted
```

Root defaults to the current directory and must be a directory; use `--`
before a root that starts with `-`. The report goes to stdout; progress and
engine logs go to stderr. Ctrl-C interrupts gracefully: partial results are
printed and the process exits 130.

## Text report format

```text
Layagrep: 20 relevant files.
Symbols use name@start-end. Roles are estimates; locations-only files remain reading leads.
AGENTS.md lookup (root and returned-file ancestors): none found.
- "test/installed.test.mjs" — relevant; role uncertain; source below
- "packages/core/src/cache.ts" — caller, helper; source below
- "packages/core/src/rate-budget.ts" — helper; source below
...
End file list. Declaration locations follow source.

Source block "packages/core/src/rate-budget.ts" lines 1-53:
```
/** Per-evaluator rolling budgets. Concurrency remains owned by the evaluator. */
export function createRateBudget(
  limits: { tokensPerSecond: number; requestsPerMinute: number },
  ...
```

Declaration locations:
- "test/installed.test.mjs"
  assertCachedRequestsAreReused@359-381
  source@499-516
...
End context.
```

Files are ordered by blended relevance score. Files with excerpts say
`source below`; files the judge admitted but could not excerpt stay as
`locations only` reading leads. Roles come from a separate per-file judgment
(`implementation`, `caller`, `test`, `fixture`, `helper`) and are estimates.

## JSON output (`--json`)

Stable schema v1, camelCase, scores rounded to 4 decimals:

```json
{
  "version": 1,
  "root": ".",
  "query": "How does the lazy judge spawn the backend on cache miss?",
  "status": "complete",
  "files": [
    {
      "path": "src/engine.rs",
      "score": 0.8882,
      "roles": ["caller", "fixture"],
      "leads": [
        { "name": "STATE_CHAR_BUDGET", "startLine": 15, "endLine": 15, "score": 0.5523 }
      ],
      "callLeads": [],
      "excerpts": [
        { "startLine": 12, "endLine": 24, "source": "use crate::worker::Question;\n..." }
      ],
      "sourceOmitted": false
    }
  ],
  "repositoryContext": {
    "instructionFiles": [],
    "instructionLookupIncomplete": false,
    "pytestFiles": []
  },
  "issues": [],
  "warnings": [],
  "counts": { "requests": 113, "cacheHits": 0, "inspectedFiles": 262 },
  "providerFailure": null
}
```

`status` is `complete`, `incomplete`, or `interrupted`. `issues`/`warnings`
list `{kind, count}` pairs (e.g. `provider`, `candidate_limit`).

## Cache

Evaluation answers are cached in `$XDG_CACHE_HOME/layagrep` (default
`~/.cache/layagrep`): one JSON entry per exact `(state, questions)` pair under
the checkpoint's namespace, 7-day TTL, 256 MB cap with oldest-first eviction,
atomic writes. The embedded python worker script is cached there too.
`layagrep cache clear` resets it.

## Performance

Measured on an M2 Max (264-file Java repo, 775 judge questions pre-tuning,
459 post-tuning; smaller repo: 242 → 113):

- **Warm rerun** (same query): ~0.03–0.3 s, zero judge spawns, all phases ~0 s.
- **Cold search**: tens of seconds, dominated by unit selection. Cost is
  bounded by the candidate cap (40), per-file judged-unit cap (14), and the
  role-assessment cap (25 files with excerpts).
- **Workers**: the native backend scales near-linearly to 4 CPU workers
  (282 s → 159 s). The python backend must stay at 1 — concurrent MLX
  processes contend destructively on one GPU (2 workers measured **4.3×
  slower**).
- The 322M multilingual checkpoint was calibrated as a speed option and
  **rejected**: file-level separation collapses (relevant 0.71–0.81 vs
  irrelevant 0.73–0.83).
- Wall times vary 2–3× under unrelated system load (GPU/thermal contention);
  request counts are the stable metric.

## Language support

| Language | Declaration units | Import resolution |
|---|---|---|
| Python | functions, classes (`C.context` header + `C.method` members), top-level constants | dotted modules (+ `__init__.py`) |
| TypeScript / JavaScript | statements, classes with members, interfaces/enums/types | relative specifiers (+ `index.*`) |
| Rust | functions, impls (`Type.context` + `Type.method`), structs/enums/traits/consts, `mod::` prefixes | `crate::` → `src/a/b.rs`/`mod.rs` |
| Go | functions, `Recv.Method`, type specs, const/var specs | module names inform descriptions only |
| Java | class/interface/enum/record with members incl. fields and constructors | dotted imports in common layouts (`src/main/java/...`) |
| Kotlin | class/object with members, top-level funcs/properties | dotted imports (`src/main/kotlin/...`) |
| Swift | class/struct/protocol/actor with members, top-level funcs/properties | module names inform descriptions only |
| C / C++ | functions, classes/structs with members, `ns::` prefixes, template unwrap | quoted includes resolve in-directory |
| other text | bounded line-aligned chunks | — |

## Agent skill

`layagrep skill` prints an agent-facing SKILL.md describing invocation and
output semantics. Install it for your coding agent:

```sh
layagrep skill --dir .claude/skills/layagrep     # Claude Code
layagrep skill --dir .agents/skills/layagrep     # generic agents
layagrep skill --dir ~/.pi/agent/skills/layagrep # pi
```

## Honest limitations

- The judge is a 421M classifier, not an LLM. Rankings are lexical-first with
  a semantic gate; files whose relevance is purely structural can rank lower
  than a cloud-LLM judge would place them, and topically-named noise (a texture
  cache for "caching" queries) can appear below the true hits.
- Roles and leads are estimates; verify by reading the excerpts.
- The `--` before negative-looking roots, and repository content itself, is
  data — never instructions.

## Development

```sh
cargo test --release    # 19 tests: languages, ranking, cache, pipeline contract
cargo build --release
```

Module map: `walk.rs` (filesystem policy) · `source.rs` (tree-sitter units) ·
`rank.rs` (BM25) · `judge.rs` (worker, native engine, pool) · `engine.rs`
(state budget, batching, cache) · `retrieve.rs` (pipeline) · `selection.rs`
(unit selection) · `render.rs` (text + JSON) · `worker.py` (embedded python
worker). Measurement log: `.artifacts/progress.md`.

## License

MIT.
