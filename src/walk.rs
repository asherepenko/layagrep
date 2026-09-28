//! Repository traversal with jevgrep's eligibility policy: respect
//! .gitignore/.ignore (plus the global git exclude file), skip hidden paths,
//! dependency/build directories, sensitive names, symlinks, binaries, and
//! non-UTF-8 content. The eligible tree is enumerated up front (metadata only,
//! no file reads) so navigation scoring stays decoupled from walking.

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const DEPENDENCY_DIRECTORIES: &[&str] = &[
    "node_modules",
    "vendor",
    "venv",
    ".venv",
    ".tox",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    "target",
    ".next",
    ".nuxt",
    ".turbo",
];

pub const SENSITIVE_NAMES: &[&str] = &[
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
];

pub const SENSITIVE_SUFFIXES: &[&str] = &[".pem", ".key", ".p12", ".pfx"];

pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_ENTRIES: usize = 100_000;
pub const MAX_IGNORE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct Policy {
    pub hidden: bool,
    pub no_ignore: bool,
    pub include_dependencies: bool,
    pub include_sensitive: bool,
}

impl Policy {
    pub fn version(&self) -> String {
        format!(
            "h{}:i{}:d{}:s{}",
            self.hidden as u8,
            self.no_ignore as u8,
            self.include_dependencies as u8,
            self.include_sensitive as u8
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildKind {
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub struct Child {
    pub name: String,
    pub kind: ChildKind,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct DirectoryInfo {
    /// Path relative to the root, "" for the root itself.
    pub path: String,
    pub children: Vec<Child>,
}

#[derive(Debug, Default)]
pub struct Tree {
    /// All eligible directories, in discovery (BFS) order.
    pub directories: Vec<DirectoryInfo>,
    pub files_seen: usize,
    pub entries_seen: usize,
    pub truncated: bool,
    /// Directory path -> index into `directories`.
    pub index: HashMap<String, usize>,
}

impl Tree {
    pub fn dir(&self, path: &str) -> Option<&DirectoryInfo> {
        self.index.get(path).map(|&i| &self.directories[i])
    }
}

#[derive(Debug)]
pub struct WalkIssue {
    pub kind: &'static str,
    pub path: String,
}

pub struct WalkResult {
    pub tree: Tree,
    pub issues: Vec<WalkIssue>,
    pub root_abs: PathBuf,
}

fn is_sensitive_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == ".env"
        || lower.starts_with(".env.")
        || SENSITIVE_NAMES.contains(&lower.as_str())
        || SENSITIVE_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix))
}

#[derive(Clone)]
struct Scope {
    /// Path prefix of this scope's directory relative to the search root,
    /// "" for the root scope. The global git exclude file matches bare names.
    prefix: String,
    git: Option<std::sync::Arc<Gitignore>>,
    search: Option<std::sync::Arc<Gitignore>>,
}

impl Scope {
    fn matches(&self, child_rel: &str, is_dir: bool) -> Option<bool> {
        let local = if self.prefix.is_empty() {
            child_rel
        } else {
            child_rel.strip_prefix(&self.prefix)?
        };
        let mut ignored = None;
        for matcher in [&self.git, &self.search] {
            let Some(matcher) = matcher else { continue };
            match matcher.matched(Path::new(local), is_dir) {
                ignore::Match::None => {}
                ignore::Match::Ignore(_) => ignored = Some(true),
                ignore::Match::Whitelist(_) => ignored = Some(false),
            }
        }
        ignored
    }
}

fn load_rules(directory: &Path, name: &str) -> Option<std::sync::Arc<Gitignore>> {
    let path = directory.join(name);
    let meta = std::fs::symlink_metadata(&path).ok()?;
    if !meta.is_file() {
        return None;
    }
    if meta.len() > MAX_IGNORE_BYTES {
        return None;
    }
    let mut builder = GitignoreBuilder::new(directory);
    match builder.add(path.as_path()) {
        Some(_) => None, // unreadable rule file: treat as absent
        None => Some(std::sync::Arc::new(builder.build().ok()?)),
    }
}

pub fn walk(root: &Path, policy: &Policy) -> WalkResult {
    let root_abs = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut tree = Tree::default();
    let mut issues: Vec<WalkIssue> = Vec::new();
    let mut global_scope = Scope {
        prefix: String::new(),
        git: None,
        search: None,
    };
    if !policy.no_ignore {
        let (global, _) = Gitignore::global();
        if !global.is_empty() {
            global_scope.git = Some(std::sync::Arc::new(global));
        }
    }

    let mut queue: std::collections::VecDeque<(PathBuf, String, Vec<Scope>)> =
        std::collections::VecDeque::new();
    queue.push_back((root_abs.clone(), String::new(), vec![global_scope]));

    while let Some((dir_abs, dir_rel, parent_scopes)) = queue.pop_front() {
        // Submodule boundary: a nested .git drops outer git scopes (root excluded).
        let mut scopes = parent_scopes;
        if dir_rel != "." && !dir_rel.is_empty() {
            if dir_abs.join(".git").symlink_metadata().is_ok() {
                scopes.retain(|scope| scope.git.is_none());
            }
        }
        if !policy.no_ignore {
            let git = load_rules(&dir_abs, ".gitignore");
            let search = load_rules(&dir_abs, ".ignore");
            let prefix = if dir_rel.is_empty() {
                String::new()
            } else {
                format!("{}/", dir_rel)
            };
            scopes.push(Scope { prefix, git, search });
        }

        let read = match std::fs::read_dir(&dir_abs) {
            Ok(read) => read,
            Err(_) => {
                if !dir_rel.is_empty() {
                    issues.push(WalkIssue { kind: "unreadable", path: dir_rel });
                }
                continue;
            }
        };
        let mut children: Vec<Child> = Vec::new();
        let mut subdirs: Vec<(PathBuf, String)> = Vec::new();
        let mut names: Vec<(String, std::io::Result<std::fs::DirEntry>)> = Vec::new();
        for entry in read {
            match entry {
                Ok(entry) => {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    names.push((name, Ok(entry)));
                }
                Err(e) => issues.push(WalkIssue {
                    kind: "unreadable",
                    path: format!("{}: {}", dir_rel, e),
                }),
            }
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, entry_result) in names {
            let entry = entry_result.expect("sorted entries are present");
            if tree.entries_seen >= MAX_ENTRIES {
                tree.truncated = true;
                break;
            }
            tree.entries_seen += 1;
            if name == ".git" {
                continue;
            }
            if !policy.hidden && name.starts_with('.') {
                continue;
            }
            if !policy.include_dependencies
                && DEPENDENCY_DIRECTORIES.contains(&name.as_str())
            {
                // Directory names only; a file named "build" stays eligible.
                let meta = entry.metadata().ok();
                if meta.as_ref().map(|m| m.is_dir()).unwrap_or(false) {
                    continue;
                }
            }
            if !policy.include_sensitive && is_sensitive_name(&name) {
                continue;
            }
            let meta = match std::fs::symlink_metadata(entry.path()) {
                Ok(meta) => meta,
                Err(_) => continue,
            };
            if meta.is_symlink() {
                continue;
            }
            let is_dir = meta.is_dir();
            if !is_dir && !meta.is_file() {
                continue; // fifo, socket, device
            }
            let child_rel = if dir_rel.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", dir_rel, name)
            };
            if !policy.no_ignore {
                let mut ignored = false;
                for scope in &scopes {
                    if let Some(value) = scope.matches(&child_rel, is_dir) {
                        ignored = value;
                    }
                }
                if ignored {
                    continue;
                }
            }
            if is_dir {
                subdirs.push((entry.path(), child_rel.clone()));
                children.push(Child { name, kind: ChildKind::Directory, size: 0 });
            } else {
                tree.files_seen += 1;
                children.push(Child { name, kind: ChildKind::File, size: meta.len() });
            }
        }

        tree.index.insert(dir_rel.clone(), tree.directories.len());
        tree.directories.push(DirectoryInfo { path: dir_rel, children });
        for (path, rel) in subdirs {
            if tree.truncated {
                break;
            }
            let scopes = scopes.clone();
            queue.push_back((path, rel, scopes));
        }
        if tree.truncated {
            issues.push(WalkIssue { kind: "resource_limit", path: ".".into() });
            break;
        }
    }

    WalkResult { tree, issues, root_abs }
}

/// Read a file snapshot with the shared eligibility checks applied to content.
pub enum SnapshotStatus {
    Ok(Snapshot),
    Excluded(&'static str),
    Issue(&'static str),
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub path: String,
    pub source: String,
    pub content_hash: String,
    pub bytes: usize,
}

pub fn read_snapshot(root: &Path, rel: &str) -> SnapshotStatus {
    let absolute = root.join(rel);
    let meta = match std::fs::symlink_metadata(&absolute) {
        Ok(meta) => meta,
        Err(_) => return SnapshotStatus::Issue("changed"),
    };
    if meta.is_symlink() || !meta.is_file() {
        return SnapshotStatus::Excluded("special_file");
    }
    if meta.len() > MAX_FILE_BYTES {
        return SnapshotStatus::Issue("resource_limit");
    }
    let bytes = match std::fs::read(&absolute) {
        Ok(bytes) => bytes,
        Err(_) => return SnapshotStatus::Issue("unreadable"),
    };
    if bytes.iter().take(8192).any(|&b| matches!(b, 0x00..=0x08 | 0x0B | 0x0E..=0x1F | 0x7F)) {
        return SnapshotStatus::Excluded("binary");
    }
    let source = match String::from_utf8(bytes) {
        Ok(source) => source,
        Err(_) => return SnapshotStatus::Excluded("invalid_utf8"),
    };
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(&source.as_bytes());
    SnapshotStatus::Ok(Snapshot {
        path: rel.to_string(),
        bytes: source.len(),
        content_hash: hex::encode(digest),
        source,
    })
}
