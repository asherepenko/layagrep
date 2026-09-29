//! layagrep — source retrieval for coding agents, judged by local Laya
//! typed-decision models. A Rust sibling of dzhng/jevgrep built on laya-mlx.

mod cache;
mod engine;
mod judge;
mod rank;
mod render;
mod retrieve;
mod selection;
mod source;
mod types;
mod walk;
mod worker;

use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

pub const SKILL_MD: &str = include_str!("SKILL.md");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Dtype {
    Float16,
    Float32,
    Bfloat16,
}

impl Dtype {
    fn as_str(&self) -> &'static str {
        match self {
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Bfloat16 => "bfloat16",
        }
    }
}

#[derive(Parser)]
#[command(
    name = "layagrep",
    about = "Find code by asking what it does — judged by a local Laya typed-decision model",
    long_about = "layagrep — source retrieval for coding agents, judged by a local Laya model.\n\nAsk a natural-language question about a repository. layagrep parses files with\ntree-sitter (Python, TS/JS, Rust, Go, Java, Kotlin, Swift, C/C++), ranks them\nlexically, gates and grades them with a local ~400M-parameter Laya encoder, and\nprints ranked files, verbatim source excerpts with line references, and\ndeclaration leads. No cloud API, no keys; everything runs locally.\n\nSibling of jevgrep (dzhng/jevgrep) with a local judge instead of the Jev cloud LLM.",
    version,
    disable_help_subcommand = true,
    arg_required_else_help = true,
    after_help = "Examples:\n  layagrep \"How are telemetry events recorded and sent?\" .\n  layagrep --json \"Where is auth checked before a handler?\" ./src\n  layagrep --engine native --workers 4 \"retry behavior on timeout?\"\n\nCommands:\n  layagrep doctor              verify the judge backend end to end\n  layagrep cache clear         clear cached judge answers\n  layagrep skill --dir .       write the agent SKILL.md\n\nEnvironment:\n  LAYAGREP_PYTHON             interpreter for the python backend (same as --python)\n\nExit codes:\n  0 complete  1 failed  2 incomplete (issues reported)  130 interrupted\n\nThe report goes to stdout; progress and engine logs go to stderr.\nEvaluation answers are cached in $XDG_CACHE_HOME/layagrep (7-day TTL, 256 MB\ncap); repeated searches skip the judge entirely."
)]
struct Cli {
    /// Natural-language repository question, e.g. "How are requests cached?"
    query: Option<String>,

    /// Search root directory (default: current directory; use -- before a root starting with '-')
    root: Option<String>,

    /// Include hidden (dot-prefixed) paths
    #[arg(long)]
    hidden: bool,

    /// Disable .gitignore/.ignore pattern matching (global git excludes still apply)
    #[arg(long)]
    no_ignore: bool,

    /// Include dependency and build directories (node_modules, target, dist, ...)
    #[arg(long)]
    include_dependencies: bool,

    /// Include known sensitive filenames (.env, *.pem, id_rsa, ...) and private-key content
    #[arg(long)]
    include_sensitive: bool,

    /// Disable answer-cache reads and writes for this run
    #[arg(long)]
    no_cache: bool,

    /// Total source-byte budget for the text report; 0 means unlimited
    #[arg(long, default_value_t = render::DEFAULT_MAX_SOURCE_BYTES)]
    max_source_bytes: usize,

    /// Judge backend: auto prefers an installed laya-mlx python package, else native
    #[arg(long, value_enum, default_value_t = EngineChoice::Auto)]
    engine: EngineChoice,

    /// Laya checkpoint: Hugging Face id or local path
    #[arg(long, default_value = judge::DEFAULT_MODEL_ID)]
    model: String,

    /// Model dtype; the native backend promotes float16/bfloat16 to float32 on CPU
    #[arg(long, value_enum, default_value_t = Dtype::Float16)]
    dtype: Dtype,

    /// Python interpreter hosting laya_mlx; default: the laya-mlx launcher's interpreter, then LAYAGREP_PYTHON, then python3
    #[arg(long)]
    python: Option<String>,

    /// Suppress stderr progress output (engine info still prints once)
    #[arg(long, short = 'q')]
    quiet: bool,

    /// Print per-file relevance scores to stderr
    #[arg(long)]
    debug_scores: bool,

    /// Emit structured JSON (schema v1: files, excerpts, leads, roles, counts) instead of the text report
    #[arg(long)]
    json: bool,

    /// Parallel judge workers (0 = auto: 1 for python, 4 for native)
    ///
    /// Measured: concurrent MLX processes contend on one GPU (2 python workers
    /// are 4.3x slower — keep 1); native CPU workers scale near-linearly to 4.
    #[arg(long, default_value_t = 0)]
    workers: usize,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EngineChoice {
    Auto,
    Native,
    Python,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify the judge backend with a synthetic question
    #[command(
        long_about = "Verify the judge backend with a synthetic question.\n\nSpawns the selected engine (first run may download the ~850 MB checkpoint from Hugging Face), runs one prediction, and checks the answer. Use it after install or when searches behave unexpectedly."
    )]
    Doctor {
        /// Judge backend: auto, python, or native
        #[arg(
            long,
            value_enum,
            default_value_t = EngineChoice::Auto,
            help = "Judge backend: auto prefers python (laya-mlx), else native"
        )]
        engine: EngineChoice,
        /// Laya checkpoint: HF id or local path
        #[arg(long, help = "Laya checkpoint: Hugging Face id or local path")]
        model: String,
        /// Model dtype; native promotes float16/bfloat16 to float32 on CPU
        #[arg(long, value_enum, default_value_t = Dtype::Float16, help = "Model dtype")]
        dtype: Dtype,
        /// Python interpreter hosting laya_mlx
        #[arg(long, help = "Python interpreter hosting laya_mlx")]
        python: Option<String>,
    },
    /// Cache management
    #[command(long_about = "Cache management.\n\nEvaluation answers live in $XDG_CACHE_HOME/layagrep (default ~/.cache/layagrep): JSON entries keyed by exact state + questions, 7-day TTL, 256 MB cap, oldest-first eviction. The resident python worker script is cached there too.")]
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    /// Print the agent skill; --dir writes SKILL.md into a directory
    #[command(
        long_about = "Print the agent skill describing layagrep usage for coding agents.\n\nWith --dir, writes SKILL.md (uppercase) into the directory — point it at your agent's skills directory, e.g. `.claude/skills/layagrep`, `.agents/skills/layagrep`, or `~/.pi/agent/skills/layagrep`."
    )]
    Skill {
        /// Directory to write SKILL.md into (stdout if omitted)
        #[arg(long, help = "Directory to write SKILL.md into (printed to stdout when omitted)")]
        dir: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum CacheAction {
    /// Clear cached evaluation answers (searches afterwards re-judge from scratch)
    Clear,
}

fn main() {
    let cli = Cli::parse();
    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::Relaxed))
        .unwrap_or_default();

    let code = match cli.command.as_ref() {
        Some(Command::Doctor { engine, model, dtype, python }) => doctor(
            to_engine(*engine),
            model,
            dtype.as_str(),
            python.as_deref(),
        ),
        Some(Command::Cache { action: CacheAction::Clear }) => cache_clear(),
        Some(Command::Skill { dir }) => skill(dir.clone()),
        None => match cli.query.clone() {
            Some(query) => search(&cli, query),
            None => {
                eprintln!("Usage: layagrep \"question\" [root]. Run layagrep --help.");
                1
            }
        },
    };
    std::process::exit(code);
}

fn to_engine(choice: EngineChoice) -> judge::EngineKind {
    match choice {
        EngineChoice::Auto => judge::EngineKind::Auto,
        EngineChoice::Native => judge::EngineKind::Native,
        EngineChoice::Python => judge::EngineKind::Python,
    }
}

fn stderr_isatty() -> bool {
    use std::io::IsTerminal;
    std::io::stderr().is_terminal()
}

fn search(cli: &Cli, query: String) -> i32 {
    let root = cli.root.clone().unwrap_or_else(|| ".".to_string());
    let root_path = PathBuf::from(&root);
    if !root_path.is_dir() {
        eprintln!("Search root must be a directory: {}", root);
        return 1;
    }
    let policy = walk::Policy {
        hidden: cli.hidden,
        no_ignore: cli.no_ignore,
        include_dependencies: cli.include_dependencies,
        include_sensitive: cli.include_sensitive,
    };
    let engine_kind = to_engine(cli.engine);
    let model_id = cli.model.clone();
    let dtype = cli.dtype.as_str().to_string();
    let python = cli.python.clone();
    let requested = cli.workers;
    let mut factory: engine::JudgeFactory = Box::new(move || {
        // Measured on M2 Max: python (MLX GPU) workers contend destructively
        // (2 workers = 4.3x slower); native CPU workers scale near-linearly.
        let workers = if requested > 0 {
            requested
        } else {
            match engine_kind {
                judge::EngineKind::Native => 4,
                _ => 1,
            }
        };
        let kind = engine_kind;
        let model = model_id.clone();
        let dtype = dtype.clone();
        let python = python.clone();
        let pool = judge::PoolJudge::spawn(workers, move |_| {
            judge::create_judge(kind, &model, &dtype, python.as_deref())
        })?;
        Ok(Box::new(pool) as Box<dyn judge::Judge>)
    });
    let cache = cache::Cache::new(worker::cache_dir(), !cli.no_cache);
    let mut engine = engine::Engine::new(&cli.model, factory, cache);
    let mut progress = retrieve::Progress::new(!cli.quiet && stderr_isatty());
    let options = retrieve::SearchOptions { policy, query: query.clone(), root, debug_scores: cli.debug_scores };
    let result = retrieve::retrieve(&options, &mut engine, &mut progress, &interrupted);
    if !cli.quiet {
        match engine.backend() {
            Some(backend) => eprintln!("[layagrep] engine: {}", backend.describe()),
            None => eprintln!("[layagrep] engine: fully cached, no judge spawned"),
        }
    }
    let mut stdout = std::io::stdout();
    let rendered = if cli.json {
        render::render_json(&result)
    } else {
        render::render_result(&result, cli.max_source_bytes)
    };
    if stdout.write_all(rendered.as_bytes()).and_then(|_| stdout.flush()).is_err() {
        return 0; // EPIPE: consumer closed early
    }
    match result.status {
        types::Status::Interrupted => 130,
        types::Status::Incomplete => 2,
        types::Status::Complete => 0,
    }
}

fn doctor(engine_kind: judge::EngineKind, model: &str, dtype: &str, python: Option<&str>) -> i32 {
    eprintln!("Starting judge backend (first run may download the checkpoint)…");
    let model_owned = model.to_string();
    let dtype_owned = dtype.to_string();
    let python_owned = python.map(str::to_string);
    let mut factory: engine::JudgeFactory =
        Box::new(move || judge::create_judge(engine_kind, &model_owned, &dtype_owned, python_owned.as_deref()));
    let mut engine = engine::Engine::new(model, factory, cache::Cache::new(worker::cache_dir(), false));
    if let Err(error) = engine.spawn_now() {
        eprintln!("Laya engine check failed: {}", error);
        return 1;
    }
    let backend = engine
        .backend()
        .map(|b| b.describe())
        .unwrap_or_else(|| "unknown".to_string());
    match engine.judge_bool(
        "Source: export function recordEvent(event) { events.push(event); }",
        "Does this source implement recording an event?",
    ) {
        Ok(probability) if probability > 0.5 => {
            println!("Laya connection verified through {} (P = {:.3}).", backend, probability);
            0
        }
        Ok(probability) => {
            eprintln!(
                "Laya returned an unexpected answer to the connection check (P = {:.3}).",
                probability
            );
            1
        }
        Err(error) => {
            eprintln!("Laya connection check failed: {}", error);
            1
        }
    }
}

fn cache_clear() -> i32 {
    let mut store = cache::Cache::new(worker::cache_dir(), true);
    match store.clear() {
        Ok(()) => {
            println!("Cache cleared.");
            0
        }
        Err(error) => {
            eprintln!("{}", error);
            1
        }
    }
}

fn skill(dir: Option<PathBuf>) -> i32 {
    match dir {
        Some(directory) => {
            if let Err(error) = std::fs::create_dir_all(&directory) {
                eprintln!("Cannot create skill directory: {}", error);
                return 1;
            }
            let path = directory.join("SKILL.md");
            if let Err(error) = std::fs::write(&path, SKILL_MD) {
                eprintln!("Cannot write skill file: {}", error);
                return 1;
            }
            println!("Skill written to {}", path.display());
            0
        }
        None => {
            print!("{}", SKILL_MD);
            0
        }
    }
}
