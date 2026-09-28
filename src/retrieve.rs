//! Retrieval pipeline v2, shaped by measured laya behavior.
//!
//! The judge is a 421M encoder that performs graded topical matching on
//! natural-language states — raw code states wash out its signal (measured:
//! LICENSE 0.86 vs cache.ts 0.87 on code previews), while short descriptions
//! separate cleanly (LICENSE 0.10, evaluator.ts 0.83). The pipeline therefore:
//!
//! 1. describes every eligible file (path, header, declaration names, imports)
//!    and asks one question per file;
//! 2. propagates scores forward along the import graph (factor 0.75, two
//!    rounds) so structurally-relevant files whose relevance lives in their
//!    dependencies score in (measured: retrieve.ts 0.37 → 0.62 via cache.ts);
//! 3. selects declaration units by description (name, signature, local calls,
//!    docstring), which separated 0.56–0.72 relevant vs 0.09–0.12 irrelevant
//!    in calibration.

use crate::engine::{state_header, Engine, EngineError};
use crate::rank::DescriptionCorpus;
use crate::selection;
use crate::source::{Inspection, LineIndex, Range};
use crate::types::*;
use crate::walk::{walk, ChildKind, Policy, Snapshot, SnapshotStatus};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

const MIN_LEXICAL: f64 = 0.06;
const SEMANTIC_FLOOR: f64 = 0.3;
const PROPAGATION_FACTOR: f64 = 0.75;
const PROPAGATION_ROUNDS: usize = 2;
const REQUEST_LIMIT: u64 = 6_000;
const MAX_CANDIDATES: usize = 40;

pub struct Progress {
    enabled: bool,
    started: Instant,
    files_total: usize,
    files_done: usize,
    stage: &'static str,
}

impl Progress {
    pub fn new(enabled: bool) -> Self {
        Progress {
            enabled,
            started: Instant::now(),
            files_total: 0,
            files_done: 0,
            stage: "describe",
        }
    }
    fn set_stage(&mut self, stage: &'static str) {
        self.stage = stage;
        self.render();
    }
    fn bump(&mut self, files: usize) {
        self.files_done += files;
        self.render();
    }
    fn render(&self) {
        if !self.enabled {
            return;
        }
        eprint!(
            "\r[layagrep] {:<10} files {}/{} elapsed {:.1}s   ",
            self.stage,
            self.files_done,
            self.files_total,
            self.started.elapsed().as_secs_f64()
        );
    }
    pub fn finish(&self) {
        if self.enabled {
            eprintln!(
                "\r[layagrep] done in {:.1}s ({} files scored)",
                self.started.elapsed().as_secs_f64(),
                self.files_done
            );
        }
    }
}

struct IssueTracker {
    issues: BTreeMap<String, u32>,
    provider_failure: Option<String>,
}

impl IssueTracker {
    fn new() -> Self {
        IssueTracker { issues: BTreeMap::new(), provider_failure: None }
    }
    fn issue(&mut self, kind: &str, count: u32) {
        *self.issues.entry(kind.to_string()).or_insert(0) += count;
    }
    fn fail(&mut self, kind: &str, message: String) {
        self.issue(kind, 1);
        if self.provider_failure.is_none() {
            self.provider_failure = Some(message);
        }
    }
}

// ---------------------------------------------------------------------------
// File descriptions and imports
// ---------------------------------------------------------------------------

struct FileFacts {
    path: String,
    description: String,
    imports: Vec<String>,
    snapshot: Snapshot,
    inspection: Option<Inspection>,
}

fn header_comment(source: &str) -> Option<String> {
    let mut lines = source.lines().filter(|l| !l.trim().is_empty());
    let mut candidates: Vec<String> = Vec::new();
    for line in lines.by_ref().take(40) {
        let trimmed = line.trim();
        if trimmed.len() > 6 {
            let text = trimmed
                .trim_start_matches('/')
                .trim_start_matches('*')
                .trim_start_matches('#')
                .trim_start_matches('!')
                .trim();
            let lower = text.to_ascii_lowercase();
            if lower.contains("license") || lower.contains("copyright") {
                continue;
            }
            if text.chars().any(|c| c.is_alphabetic()) {
                candidates.push(text.chars().take(120).collect::<String>());
            }
        }
        if candidates.len() >= 1 && !trimmed.starts_with("//") && !trimmed.starts_with('#') {
            break;
        }
    }
    candidates.into_iter().next()
}

/// Module specifiers imported by a file, as written.
fn import_specifiers(snapshot: &Snapshot, inspection: Option<&Inspection>) -> Vec<String> {
    let mut specs = Vec::new();
    let is_python = snapshot.path.ends_with(".py") || snapshot.path.ends_with(".pyi");
    let is_rust = snapshot.path.ends_with(".rs");
    let is_go = snapshot.path.ends_with(".go");
    let is_java = snapshot.path.ends_with(".java");
    let is_kotlin = snapshot.path.ends_with(".kt") || snapshot.path.ends_with(".kts");
    let is_swift = snapshot.path.ends_with(".swift");
    let is_cpp = matches!(
        snapshot.path.rsplit('.').next().unwrap_or(""),
        "cpp" | "cc" | "cxx" | "c" | "h" | "hpp" | "hh" | "hxx" | "ipp"
    );
    for line in snapshot.source.lines().take(400) {
        let trimmed = line.trim();
        if is_python {
            if let Some(rest) = trimmed.strip_prefix("from ") {
                if let Some((module, _)) = rest.split_once(" import ") {
                    specs.push(module.trim().to_string());
                }
            } else if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.split(',').next().unwrap_or("").trim();
                let module = module.split(" as ").next().unwrap_or(module).trim();
                if !module.is_empty() {
                    specs.push(module.to_string());
                }
            }
        } else if is_rust {
            if let Some(rest) = trimmed.strip_prefix("use ") {
                let module = rest.split(';').next().unwrap_or("").trim();
                let module = module.split(" as ").next().unwrap_or(module).trim();
                let module = module.trim_start_matches("r#");
                let module = module.split('{').next().unwrap_or(module).trim();
                if !module.is_empty() {
                    specs.push(module.to_string());
                }
            }
        } else if is_go {
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.trim().trim_matches('"');
                if !module.is_empty() {
                    specs.push(module.to_string());
                }
            }
        } else if is_java || is_kotlin {
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest
                    .trim_start_matches("static ")
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    specs.push(module.to_string());
                }
            }
        } else if is_swift {
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.split(';').next().unwrap_or("").trim();
                if !module.is_empty() {
                    specs.push(module.to_string());
                }
            }
        } else if is_cpp {
            if trimmed.starts_with("#include") {
                if let Some(position) = trimmed.find('"') {
                    if let Some(end) = trimmed[position + 1..].find('"') {
                        let module = &trimmed[position + 1..position + 1 + end];
                        if !module.is_empty() {
                            specs.push(module.to_string());
                        }
                    }
                }
            }
        } else {
            let extracted = |text: &str| -> Option<String> {
                let quote = text.chars().rev().find(|c| *c == '"' || *c == '\'')?;
                let start = text.rfind(quote)? + 1;
                let end = text.len() - text.chars().rev().collect::<String>().find(quote)? - 1;
                if end > start && end <= text.len() {
                    Some(text[start..end].to_string())
                } else {
                    None
                }
            };
            let mut found = None;
            if trimmed.starts_with("import ") {
                if let Some(position) = trimmed.rfind(" from ") {
                    found = extracted(&trimmed[position + 6..]);
                }
                if found.is_none() {
                    // import "module";
                    found = extracted(trimmed);
                }
            } else if trimmed.starts_with("export ") && trimmed.contains(" from ") {
                if let Some(position) = trimmed.rfind(" from ") {
                    found = extracted(&trimmed[position + 6..]);
                }
            } else if trimmed.starts_with("const ") || trimmed.starts_with("let ") || trimmed.starts_with("var ") {
                if let Some(position) = trimmed.find("require(") {
                    found = extracted(&trimmed[position + 8..]);
                }
            }
            if let Some(spec) = found {
                specs.push(spec);
            }
        }
    }
    let _ = inspection;
    specs.truncate(24);
    specs
}

/// Resolve a module specifier to a repository file, or None.
fn resolve_import(spec: &str, from_path: &str, files: &HashSet<String>) -> Option<String> {
    let from_dir = from_path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let mut candidates: Vec<String> = Vec::new();
    if spec.starts_with('.') {
        // Relative module path.
        let joined = if let Some(stripped) = spec.strip_prefix("./") {
            format!("{}/{}", from_dir, stripped)
        } else {
            // ../ chains
            let mut dir_parts: Vec<&str> = if from_dir.is_empty() {
                Vec::new()
            } else {
                from_dir.split('/').collect()
            };
            let mut rest = spec;
            while let Some(stripped) = rest.strip_prefix("../") {
                dir_parts.pop();
                rest = stripped;
            }
            let rest = rest.strip_prefix("./").unwrap_or(rest);
            let prefix = dir_parts.join("/");
            if prefix.is_empty() {
                rest.to_string()
            } else {
                format!("{}/{}", prefix, rest)
            }
        };
        let joined = joined.trim_start_matches('/').replace("//", "/");
        candidates.push(joined.clone());
        for ext in [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".pyi", ".md", ".json"] {
            candidates.push(format!("{}{}", joined, ext));
        }
        for index in ["index.ts", "index.tsx", "index.js", "__init__.py"] {
            candidates.push(format!("{}/{}", joined, index));
        }
    } else if spec.matches('.').count() >= 1
        && !spec.contains('/')
        && !spec.contains("::")
        && !spec.contains('"')
    {
        // Java/Kotlin dotted imports: a.b.C -> a/b/C.{java,kt} in common layouts.
        let slashed = spec.replace('.', "/");
        for ext in [".java", ".kt"] {
            let mut paths = vec![format!("{}{}", slashed, ext)];
            for base in ["src", "src/main/java", "src/main/kotlin"] {
                let candidate = format!("{}/{}{}", base, slashed, ext);
                if !paths.contains(&candidate) {
                    paths.push(candidate);
                }
            }
            if let Some(hit) = paths.into_iter().find(|p| p != from_path && files.contains(p)) {
                return Some(hit);
            }
        }
        return None;
    } else if spec.starts_with("crate::") {
        let slashed = spec
            .trim_start_matches("crate::")
            .split("::")
            .filter(|part| *part != "self")
            .collect::<Vec<_>>()
            .join("/");
        for base in ["src", "."] {
            candidates.push(format!("{}/{}/mod.rs", base, slashed));
            candidates.push(format!("{}/{}.rs", base, slashed));
        }
    } else {
        // Python-style dotted module.
        let slashed = spec.trim_start_matches('.').replace('.', "/");
        for tail in [&slashed, &format!("{}.py", slashed), &format!("{}/__init__.py", slashed)] {
            candidates.push(tail.clone());
        }
    }
    if candidates.is_empty() {
        // Quoted C/C++ includes resolve against the including file's directory.
        let from_dir = from_path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
        let joined = if from_dir.is_empty() {
            spec.to_string()
        } else {
            format!("{}/{}", from_dir, spec)
        };
        if joined != from_path && files.contains(&joined) {
            return Some(joined);
        }
    }
    candidates.into_iter().find(|candidate| candidate != from_path && files.contains(candidate))
}

fn describe_file(snapshot: &Snapshot, inspection: &Option<Inspection>) -> String {
    let lines = snapshot.source.matches('\n').count() + 1;
    let mut parts = vec![format!("File: {} ({} lines)", snapshot.path, lines)];
    if let Some(header) = header_comment(&snapshot.source) {
        parts.push(format!("Header: {}", header));
    }
    if let Some(inspection) = inspection {
        let names: Vec<String> = inspection
            .units
            .iter()
            .filter(|unit| !unit.name.ends_with(".context") && unit.name != "source")
            .map(|unit| unit.name.clone())
            .take(12)
            .collect();
        if !names.is_empty() {
            parts.push(format!("Declarations: {}", names.join(", ")));
        }
    }
    let specs = import_specifiers(snapshot, inspection.as_ref());
    if !specs.is_empty() {
        let shown: Vec<String> = specs.iter().take(6).cloned().collect();
        parts.push(format!("Imports: {}", shown.join(", ")));
    }
    if parts.len() == 1 {
        // No declarations and no imports: quote the opening content.
        let content: String = snapshot.source.chars().take(200).collect();
        parts.push(format!("Content: {}", content.replace('\n', " ")));
    }
    let description = parts.join("\n");
    // Hard cap: per-question cost grows steeply with state length.
    if description.len() > 600 {
        let mut end = 600;
        while end > 0 && !description.is_char_boundary(end) {
            end -= 1;
        }
        description[..end].to_string()
    } else {
        description
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

pub struct SearchOptions {
    pub policy: Policy,
    pub query: String,
    pub root: String,
    pub debug_scores: bool,
}

/// The judge is a lexical-grader: the question must echo the query topic
/// itself, not refer to it abstractly (measured in calibration probes).
fn file_question(query: &str) -> String {
    format!(
        "Does this file contain code that implements, calls, or tests: {}? \
Judge from the file path, its declared names, its header, and its imports.",
        query
    )
}

pub fn retrieve(
    options: &SearchOptions,
    engine: &mut Engine,
    progress: &mut Progress,
    interrupted: &dyn Fn() -> bool,
) -> RetrievalResult {
    let mut issues = IssueTracker::new();
    let phase_t0 = Instant::now();
    let root_abs = Path::new(&options.root).to_path_buf();
    let walked = walk(&root_abs, &options.policy);
    let t_walk = phase_t0.elapsed();
    for issue in &walked.issues {
        issues.issue(issue.kind, 1);
    }
    if walked.tree.truncated {
        issues.issue("resource_limit", 1);
    }

    // ---- Phase 1: describe every file, collect import edges ----
    let mut facts_by_path: HashMap<String, FileFacts> = HashMap::new();
    let mut edges: HashMap<String, Vec<String>> = HashMap::new(); // importer -> importees
    let file_set: HashSet<String> = walked
        .tree
        .directories
        .iter()
        .flat_map(|dir| {
            dir.children
                .iter()
                .filter(|child| child.kind == ChildKind::File)
                .map(|child| {
                    if dir.path.is_empty() {
                        child.name.clone()
                    } else {
                        format!("{}/{}", dir.path, child.name)
                    }
                })
        })
        .collect();
    progress.files_total = file_set.len();

    for directory in &walked.tree.directories {
        for child in &directory.children {
            if child.kind != ChildKind::File {
                continue;
            }
            let path = if directory.path.is_empty() {
                child.name.clone()
            } else {
                format!("{}/{}", directory.path, child.name)
            };
            match crate::walk::read_snapshot(&root_abs, &path) {
                SnapshotStatus::Ok(snapshot) => {
                    let inspection = if snapshot.bytes <= crate::source::MAX_PARSE_BYTES {
                        Some(crate::source::inspect(&snapshot))
                    } else {
                        None
                    };
                    let imports: Vec<String> = import_specifiers(&snapshot, inspection.as_ref())
                        .iter()
                        .filter_map(|spec| resolve_import(spec, &path, &file_set))
                        .collect();
                    let description = describe_file(&snapshot, &inspection);
                    edges.insert(path.clone(), imports);
                    facts_by_path.insert(
                        path.clone(),
                        FileFacts { path, description, imports: Vec::new(), snapshot, inspection },
                    );
                }
                SnapshotStatus::Excluded(_) => {}
                SnapshotStatus::Issue(kind) => issues.issue(kind, 1),
            }
        }
        progress.bump(0);
    }

    let t_describe = phase_t0.elapsed();
    // ---- Phase 2: lexical rank + semantic gate ----
    progress.set_stage("score");
    let mut scores: HashMap<String, f64> = HashMap::new();
    let question = file_question(&options.query);
    let mut paths: Vec<&String> = facts_by_path.keys().collect();
    paths.sort();
    let corpus = {
        let documents: Vec<String> = paths
            .iter()
            .map(|path| facts_by_path[*path].description.clone())
            .collect();
        DescriptionCorpus::new(paths.iter().map(|p| (*p).clone()).collect(), documents)
    };
    let lexical = corpus.rank(&options.query);
    let lexical_by_path: HashMap<&String, f64> =
        paths.iter().zip(lexical.iter()).map(|(path, score)| (*path, *score)).collect();
    // Only files with some query-term overlap reach the judge.
    for path in paths.iter().filter(|path| lexical_by_path[*path] > 0.02) {
        if interrupted() || engine.requests > REQUEST_LIMIT {
            issues.issue("interrupted", 1);
            break;
        }
        let facts = &facts_by_path[*path];
        let state = format!("{}\n{}", state_header(&options.query), facts.description);
        let questions = vec![(
            "q".to_string(),
            crate::worker::Question::Noul { instructions: question.clone() },
        )];
        match engine.evaluate(&crate::engine::Evaluation { state: &state, questions: &questions }) {
            Ok(answers) => {
                let score = answers.get("q").copied().unwrap_or(0.0);
                if options.debug_scores {
                    let lex = lexical_by_path[*path];
                    eprintln!("[score] laya={:.3} lex={:.3} {}", score, lex, path);
                }
                scores.insert((*path).clone(), score);
            }
            Err(EngineError(message)) => {
                issues.fail("provider", message);
                scores.insert((*path).clone(), 0.0);
            }
        }
        progress.bump(1);
    }

    let t_score = phase_t0.elapsed();
    // ---- Phase 3: propagate along import edges (lexical evidence spreads
    // through dependencies: a file importing a hot module shares its topic) ----
    let mut lexical_graph: HashMap<String, f64> =
        lexical_by_path.iter().map(|(path, score)| ((*path).clone(), *score)).collect();
    for _ in 0..PROPAGATION_ROUNDS {
        let mut updated: Vec<(String, f64)> = Vec::new();
        for (importer, importees) in &edges {
            let base = lexical_graph.get(importer).copied().unwrap_or(0.0);
            let best = importees
                .iter()
                .filter_map(|importee| lexical_graph.get(importee))
                .fold(0.0_f64, |acc: f64, value: &f64| acc.max(*value));
            let boosted = base.max(best * PROPAGATION_FACTOR);
            if boosted > base + 1e-9 {
                updated.push((importer.clone(), boosted));
            }
        }
        for (path, score) in updated {
            lexical_graph.insert(path, score);
        }
    }

    // ---- Phase 4: blend lexical rank with the semantic gate ----
    progress.set_stage("select");
    let mut ranked: Vec<(String, f64)> = facts_by_path
        .keys()
        .filter_map(|path| {
            let lexical_score = lexical_graph.get(path).copied().unwrap_or(0.0);
            if lexical_score < MIN_LEXICAL {
                return None;
            }
            let semantic = scores.get(path).copied().unwrap_or(0.0);
            if semantic < SEMANTIC_FLOOR {
                // The judge firmly rejects the topic match (LICENSE, lockfiles).
                return None;
            }
            let blend = 0.7 * lexical_score + 0.3 * semantic;
            Some((path.clone(), blend))
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    if options.debug_scores {
        for (path, blend) in ranked.iter().take(MAX_CANDIDATES + 5) {
            eprintln!("[blend] {:.3} {}", blend, path);
        }
    }
    if ranked.len() > MAX_CANDIDATES {
        issues.issue("candidate_limit", (ranked.len() - MAX_CANDIDATES) as u32);
        ranked.truncate(MAX_CANDIDATES);
    }

    let mut files: Vec<FileEvidence> = Vec::new();
    let mut declarations: HashMap<String, Vec<(String, Range)>> = HashMap::new();
    for (path, score) in &ranked {
        if interrupted() {
            break;
        }
        let Some(facts) = facts_by_path.get(path) else { continue };
        let (outcome, selection_issues) = selection::select_file(
            &facts.snapshot,
            facts.inspection.as_ref(),
            &options.query,
            engine,
        );
        for (kind, count) in selection_issues {
            issues.issue(kind, count);
        }
        if let Some(inspection) = &facts.inspection {
            declarations.insert(
                path.clone(),
                inspection.units.iter().map(|u| (u.name.clone(), u.range)).collect(),
            );
        }
        let call_leads = if is_python_path(path) {
            python_call_leads(&facts.snapshot, &outcome.selected, facts.inspection.as_ref())
        } else {
            Vec::new()
        };
        files.push(FileEvidence {
            path: path.clone(),
            score: *score,
            roles: Vec::new(),
            priority: None,
            leads: outcome.leads,
            call_leads,
            excerpts: outcome.excerpts,
            source_omitted: false,
        });
    }

    let t_select = phase_t0.elapsed();
    // ---- Phase 5: role assessment ----
    progress.set_stage("assess");
    for file in &mut files {
        if interrupted() {
            break;
        }
        let Some(facts) = facts_by_path.get(&file.path) else { continue };
        let mut description = facts.description.clone();
        if description.len() > 320 {
            let mut end = 320;
            while end > 0 && !description.is_char_boundary(end) {
                end -= 1;
            }
            description.truncate(end);
        }
        let state = format!("{}\n{}", state_header(&options.query), description);
        let questions = assessment_questions(&options.query);
        match engine.evaluate(&crate::engine::Evaluation { state: &state, questions: &questions }) {
            Ok(answers) => {
                for role in ["implementation", "caller", "test", "fixture", "helper"] {
                    if answers.get(role).copied().unwrap_or(0.0) > 0.65 {
                        file.roles.push(role);
                    }
                }
            }
            Err(EngineError(message)) => issues.fail("provider", message),
        }
    }

    let t_assess = phase_t0.elapsed();
    // ---- Phase 6: repository context ----
    let repository_context =
        build_repository_context(&walked.tree, &files, &declarations, &root_abs);

    progress.finish();
    if options.debug_scores {
        eprintln!(
            "[stats] requests={} cache_hits={} files={} | walk={:.1}s describe={:.1}s score={:.1}s select={:.1}s assess={:.1}s",
            engine.requests,
            engine.cache_hits(),
            files.len(),
            t_walk.as_secs_f64(),
            (t_describe - t_walk).as_secs_f64(),
            (t_score - t_describe).as_secs_f64(),
            (t_select - t_score).as_secs_f64(),
            (t_assess - t_select).as_secs_f64(),
        );
    }
    let status = if interrupted() {
        Status::Interrupted
    } else if !issues.issues.is_empty() {
        Status::Incomplete
    } else {
        Status::Complete
    };
    let inspected = facts_by_path.len();
    RetrievalResult {
        root: options.root.clone(),
        query: options.query.clone(),
        status,
        files,
        repository_context,
        issues: issues.issues.into_iter().collect(),
        warnings: engine.cache_issues(),
        counts: Counts {
            requests: engine.requests,
            cache_hits: engine.cache_hits(),
            inspected_files: inspected,
        },
        provider_failure: issues.provider_failure,
    }
}

fn is_python_path(path: &str) -> bool {
    path.ends_with(".py") || path.ends_with(".pyi")
}

const ROLE_INSTRUCTIONS: &[(&str, &str)] = &[
    ("implementation", "contains the code that directly implements: {}"),
    ("caller", "calls or integrates the code that implements: {}"),
    ("test", "contains executable tests of: {}"),
    ("fixture", "provides data or helpers that exercise: {}"),
    ("helper", "provides supporting behavior needed to understand: {}"),
];

fn assessment_questions(query: &str) -> Vec<(String, crate::worker::Question)> {
    ROLE_INSTRUCTIONS
        .iter()
        .map(|(name, template)| {
            (
                name.to_string(),
                crate::worker::Question::Noul {
                    instructions: format!("Does this file {}?", template.replacen("{}", query, 1)),
                },
            )
        })
        .collect()
}

/// Local call detection for Python: names defined in this file, called from
/// within selected ranges.
fn python_call_leads(
    snapshot: &Snapshot,
    selected: &[Range],
    inspection: Option<&Inspection>,
) -> Vec<CallLead> {
    let Some(inspection) = inspection else { return Vec::new() };
    if selected.is_empty() {
        return Vec::new();
    }
    let index = LineIndex::new(&snapshot.source);
    let defs: Vec<(&str, &Range)> = inspection
        .units
        .iter()
        .filter(|unit| !unit.name.ends_with(".context") && unit.name != "source")
        .map(|unit| {
            let bare = unit.name.rsplit('.').next().unwrap_or(&unit.name);
            (bare, &unit.range)
        })
        .collect();
    let mut leads = Vec::new();
    let mut seen = HashSet::new();
    for range in selected {
        for line_number in range.start_line..=range.end_line {
            let line = crate::source::lines_text(
                &snapshot.source,
                &index,
                Range { start_line: line_number, end_line: line_number },
            );
            for (name, def_range) in &defs {
                if line_number >= def_range.start_line && line_number <= def_range.end_line {
                    continue; // definition line, not a call
                }
                if line.contains(&format!("{}(", name)) {
                    let caller = inspection
                        .units
                        .iter()
                        .filter(|unit| {
                            line_number >= unit.range.start_line && line_number <= unit.range.end_line
                        })
                        .max_by_key(|unit| unit.range.start_line)
                        .map(|unit| unit.name.clone())
                        .unwrap_or_else(|| "module".to_string());
                    if seen.insert((caller.clone(), (*name).to_string(), line_number)) {
                        leads.push(CallLead {
                            caller,
                            name: (*name).to_string(),
                            range: Range { start_line: line_number, end_line: line_number },
                        });
                    }
                }
            }
        }
    }
    leads
}

fn build_repository_context(
    _tree: &crate::walk::Tree,
    files: &[FileEvidence],
    declarations: &HashMap<String, Vec<(String, Range)>>,
    root: &Path,
) -> RepoContext {
    let mut directories: HashSet<String> = HashSet::new();
    directories.insert(String::new());
    for file in files {
        let mut dir =
            file.path.rsplit_once('/').map(|(dir, _)| dir.to_string()).unwrap_or_default();
        loop {
            directories.insert(dir.clone());
            match dir.rsplit_once('/') {
                Some((parent, _)) => dir = parent.to_string(),
                None => break,
            }
        }
    }
    let mut instruction_files = Vec::new();
    let mut incomplete = false;
    for directory in &directories {
        let name = if directory.is_empty() {
            "AGENTS.md".to_string()
        } else {
            format!("{}/AGENTS.md", directory)
        };
        match std::fs::symlink_metadata(root.join(&name)) {
            Ok(meta) if meta.is_file() => instruction_files.push(name),
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => incomplete = true,
        }
    }
    instruction_files.sort();

    let mut pytest_files = Vec::new();
    for file in files {
        if !file.path.ends_with(".py") || file.excerpts.is_empty() {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(root.join(&file.path)) else {
            continue;
        };
        let has_pytest = source.lines().any(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("import pytest") || trimmed.starts_with("from pytest ")
        });
        if !has_pytest {
            continue;
        }
        let has_test_unit = declarations
            .get(&file.path)
            .map(|units| {
                units.iter().any(|(name, range)| {
                    name.rsplit('.').next().unwrap_or(name).starts_with("test_")
                        && file.excerpts.iter().any(|excerpt| {
                            range.start_line >= excerpt.range.start_line
                                && range.end_line <= excerpt.range.end_line
                        })
                })
            })
            .unwrap_or(false);
        if has_test_unit {
            pytest_files.push(file.path.clone());
        }
    }

    RepoContext { instruction_files, instruction_lookup_incomplete: incomplete, pytest_files }
}
