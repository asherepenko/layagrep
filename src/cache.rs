//! Answer-only evaluation cache, mirrored from jevgrep's design: sha256-keyed
//! JSON entries with a TTL and a total size cap, written atomically.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCHEMA: u64 = 1;
const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ENTRY_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheIssue {
    Unavailable,
    Corrupt,
    Limit,
}

impl CacheIssue {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheIssue::Unavailable => "cache_unavailable",
            CacheIssue::Corrupt => "cache_corrupt",
            CacheIssue::Limit => "cache_limit",
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq)]
pub struct Namespace {
    pub model: String,
    pub dtype: String,
    pub protocol: String,
    pub prompt_version: String,
}

pub struct Cache {
    directory: PathBuf,
    enabled: bool,
    pub hits: u64,
    pub misses: u64,
    issues: HashMap<CacheIssue, u32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Entry {
    schema: u64,
    created_at: u64,
    answers: HashMap<String, f64>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl Cache {
    pub fn new(directory: PathBuf, enabled: bool) -> Self {
        Cache {
            directory,
            enabled,
            hits: 0,
            misses: 0,
            issues: HashMap::new(),
        }
    }

    fn warn(&mut self, issue: CacheIssue) {
        *self.issues.entry(issue).or_insert(0) += 1;
    }

    pub fn issues(&self) -> Vec<(String, u32)> {
        self.issues
            .iter()
            .map(|(kind, count)| (kind.as_str().to_string(), *count))
            .collect()
    }

    fn entries_dir(&self) -> PathBuf {
        self.directory.join("entries")
    }

    pub fn key(namespace: &Namespace, request: &serde_json::Value) -> String {
        use sha2::{Digest, Sha256};
        let material = serde_json::json!([SCHEMA, namespace, request]);
        let digest = Sha256::digest(serde_json::to_string(&material).unwrap_or_default().as_bytes());
        hex::encode(digest)
    }

    fn valid_answers(value: &serde_json::Value) -> bool {
        value.as_object().map(|map| {
            map.values().all(|v| v.as_f64().map(|f| f.is_finite()).unwrap_or(false))
        }).unwrap_or(false)
    }

    pub fn get(&mut self, key: &str) -> Option<HashMap<String, f64>> {
        if !self.enabled {
            self.misses += 1;
            return None;
        }
        let path = self.entries_dir().join(format!("{}.json", key));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                self.misses += 1;
                return None;
            }
        };
        if bytes.len() as u64 > MAX_ENTRY_BYTES {
            self.warn(CacheIssue::Corrupt);
            self.misses += 1;
            return None;
        }
        let value: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => {
                self.warn(CacheIssue::Corrupt);
                self.misses += 1;
                return None;
            }
        };
        if value.get("schema").and_then(|v| v.as_u64()) != Some(SCHEMA) {
            self.warn(CacheIssue::Corrupt);
            self.misses += 1;
            return None;
        }
        let created_at = value.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
        let age = now_unix().saturating_sub(created_at);
        if age >= TTL.as_millis() as u64 {
            self.misses += 1;
            return None;
        }
        let answers = match value.get("answers") {
            Some(answers) if Self::valid_answers(answers) => answers.clone(),
            _ => {
                self.warn(CacheIssue::Corrupt);
                self.misses += 1;
                return None;
            }
        };
        self.hits += 1;
        serde_json::from_value(answers).ok()
    }

    pub fn put(&mut self, key: &str, answers: &HashMap<String, f64>) {
        if !self.enabled {
            return;
        }
        if !answers.values().all(|v| v.is_finite()) {
            self.warn(CacheIssue::Corrupt);
            return;
        }
        let entry = Entry {
            schema: SCHEMA,
            created_at: now_unix(),
            answers: answers.clone(),
        };
        let payload = match serde_json::to_vec(&entry) {
            Ok(payload) => payload,
            Err(_) => return,
        };
        if payload.len() as u64 > MAX_ENTRY_BYTES {
            self.warn(CacheIssue::Limit);
            return;
        }
        let entries = self.entries_dir();
        if std::fs::create_dir_all(&entries).is_err() {
            self.warn(CacheIssue::Unavailable);
            return;
        }
        let tmp = entries.join(format!(".pending-{}", uuid_like()));
        if std::fs::write(&tmp, &payload).is_err() {
            self.warn(CacheIssue::Unavailable);
            return;
        }
        let destination = entries.join(format!("{}.json", key));
        if std::fs::rename(&tmp, &destination).is_err() {
            self.warn(CacheIssue::Unavailable);
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        self.trim(&entries);
    }

    /// Streaming retention bound: evict oldest-mtime entries past the cap.
    fn trim(&mut self, entries: &Path) {
        let read = match std::fs::read_dir(entries) {
            Ok(read) => read,
            Err(_) => return,
        };
        let mut sizes: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        let mut total: u64 = 0;
        for entry in read.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".json") && !name.starts_with(".pending-") {
                continue;
            }
            let meta = match std::fs::metadata(&path) {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            if !meta.is_file() {
                continue;
            }
            let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
            if name.starts_with(".pending-") {
                // Abandoned publications get an hour before cleanup.
                if SystemTime::now()
                    .duration_since(mtime)
                    .map(|d| d > Duration::from_secs(3600))
                    .unwrap_or(false)
                {
                    let _ = std::fs::remove_file(&path);
                }
                continue;
            }
            total += meta.len();
            sizes.push((path, meta.len(), mtime));
        }
        if total <= MAX_BYTES {
            return;
        }
        sizes.sort_by_key(|(_, _, mtime)| *mtime);
        for (path, size, _) in sizes {
            if total <= MAX_BYTES {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
            }
        }
    }

    /// Detach the current generation atomically; later writers recreate it.
    pub fn clear(&mut self) -> Result<(), String> {
        let entries = self.entries_dir();
        if let Err(e) = std::fs::create_dir_all(&self.directory) {
            return Err(format!("cache directory unavailable: {}", e));
        }
        match std::fs::rename(&entries, self.directory.join(format!(".cleared-{}", uuid_like()))) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("could not clear cache: {}", e)),
        }
        self.cleanup_detached();
        Ok(())
    }

    fn cleanup_detached(&self) {
        if let Ok(read) = std::fs::read_dir(&self.directory) {
            for entry in read.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with(".cleared-") {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }
}

fn uuid_like() -> String {
    use sha2::{Digest, Sha256};
    let material = format!(
        "{}:{}:{}",
        std::process::id(),
        now_unix(),
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0)
    );
    hex::encode(&Sha256::digest(material.as_bytes())[..16])
}
