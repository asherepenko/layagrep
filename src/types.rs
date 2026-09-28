//! Shared retrieval result model, mirroring jevgrep's `RetrievalResult`.

use crate::source::Range;

#[derive(Debug, Clone)]
pub struct ReadingLead {
    pub name: String,
    pub range: Range,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CallLead {
    pub caller: String,
    pub name: String,
    pub range: Range,
}

#[derive(Debug, Clone)]
pub struct Excerpt {
    pub range: Range,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct FileEvidence {
    pub path: String,
    pub score: f64,
    pub roles: Vec<&'static str>,
    pub priority: Option<f64>,
    pub leads: Vec<ReadingLead>,
    pub call_leads: Vec<CallLead>,
    pub excerpts: Vec<Excerpt>,
    pub source_omitted: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RepoContext {
    pub instruction_files: Vec<String>,
    pub instruction_lookup_incomplete: bool,
    pub pytest_files: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Complete,
    Incomplete,
    Interrupted,
}

#[derive(Debug, Clone, Default)]
pub struct Counts {
    pub requests: u64,
    pub cache_hits: u64,
    pub inspected_files: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub root: String,
    pub query: String,
    pub status: Status,
    pub files: Vec<FileEvidence>,
    pub repository_context: RepoContext,
    pub issues: Vec<(String, u32)>,
    pub warnings: Vec<(String, u32)>,
    pub counts: Counts,
    pub provider_failure: Option<String>,
}
