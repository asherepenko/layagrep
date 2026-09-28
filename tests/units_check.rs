#[path = "../src/walk.rs"]
mod walk;
#[path = "../src/source.rs"]
mod source;

use source::inspect;
use walk::Snapshot;

fn snapshot_of(path: &str, text: &str) -> Snapshot {
    Snapshot {
        path: path.into(),
        source: text.to_string(),
        content_hash: String::new(),
        bytes: text.len(),
    }
}

fn names(snapshot: &Snapshot) -> Vec<String> {
    inspect(snapshot)
        .units
        .iter()
        .map(|u| format!("{}@{}-{}", u.name, u.range.start_line, u.range.end_line))
        .collect()
}

#[test]
fn export_naming() {
    let text = "import { x } from \"y\";\n\nexport type Thing = { a: string };\n\nexport function makeThing(): Thing {\n  return { a: \"b\" };\n}\n";
    let units = names(&snapshot_of("mod.ts", text));
    assert!(units.iter().any(|n| n.starts_with("makeThing")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Thing")), "units: {:?}", units);
}

#[test]
fn rust_units() {
    let text = r#"//! Cache module.

use crate::engine::Engine;

/// Cached answers.
struct AnswerStore { inner: Vec<u8> }

impl AnswerStore {
    fn get(&self, key: &str) -> Option<u32> {
        None
    }
}

pub fn build_cache(path: &str) -> AnswerStore {
    AnswerStore { inner: Vec::new() }
}
"#;
    let units = names(&snapshot_of("cache.rs", text));
    assert!(units.iter().any(|n| n.starts_with("AnswerStore.context")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("AnswerStore.get")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("build_cache")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("AnswerStore")), "units: {:?}", units);
}

#[test]
fn go_units() {
    let text = r#"package store

// Get returns a cached value.
func (s *Store) Get(key string) (string, bool) {
    return "", false
}

func NewStore() *Store {
    return &Store{}
}

type Store struct {
    data map[string]string
}

var ErrMissing = errors.New("missing")
"#;
    let units = names(&snapshot_of("store.go", text));
    assert!(units.iter().any(|n| n.starts_with("Store.Get")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("NewStore")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Store@")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("ErrMissing")), "units: {:?}", units);
}

#[test]
fn python_units() {
    let text = "\"\"\"Auth helpers.\"\"\"\n\nimport hashlib\n\n\ndef verify(user, pw):\n    return True\n\n\nclass Session:\n    def login(self, user):\n        return user\n";
    let units = names(&snapshot_of("auth.py", text));
    assert!(units.iter().any(|n| n.starts_with("verify")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Session.context")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Session.login")), "units: {:?}", units);
}
