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

#[test]
fn java_units() {
    let text = "package a.b;\nimport java.util.List;\n\npublic class Greeter {\n    private String name;\n\n    public Greeter(String name) {\n        this.name = name;\n    }\n\n    public String greet() {\n        return \"hi\";\n    }\n}\n\ninterface Hello {\n    void say();\n}\n";
    let units = names(&snapshot_of("Greeter.java", text));
    assert!(units.iter().any(|n| n.starts_with("Greeter.context")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Greeter.greet")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Greeter.name")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Hello")), "units: {:?}", units);
}

#[test]
fn swift_units() {
    let text = "import Foundation\n\npublic struct Point {\n    var x: Int = 0\n    func magnitude() -> Int {\n        return x * x\n    }\n}\n\nclass Api {\n    func fetch() -> Data? {\n        return nil\n    }\n}\n\nfunc topLevel(x: Int) -> Int {\n    return x\n}\n";
    let units = names(&snapshot_of("Point.swift", text));
    assert!(units.iter().any(|n| n.starts_with("Point.context")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Point.magnitude")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Api.fetch")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("topLevel")), "units: {:?}", units);
}

#[test]
fn kotlin_units() {
    let text = "package a.b\n\nclass Greeter(val name: String) {\n    fun greet(): String = \"hi\"\n}\n\nobject Singleton {\n    fun instance(): Singleton = this\n}\n\nfun topLevel(x: Int): Int = x\n";
    let units = names(&snapshot_of("Greeter.kt", text));
    assert!(units.iter().any(|n| n.starts_with("Greeter.greet")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("Singleton.instance")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("topLevel")), "units: {:?}", units);
}

#[test]
fn cpp_units() {
    let text = "#include \"local.h\"\n\nnamespace engine {\n\nclass Cache {\npublic:\n    int get(const char* key);\nprivate:\n    int size_ = 0;\n};\n\nint Cache::get(const char* key) {\n    return 0;\n}\n\ntemplate <typename T>\nT identity(T x) { return x; }\n\n}\n";
    let units = names(&snapshot_of("cache.cpp", text));
    assert!(units.iter().any(|n| n.contains("Cache.context")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.starts_with("engine.Cache.get") || n.starts_with("Cache.get")), "units: {:?}", units);
    assert!(units.iter().any(|n| n.contains("identity")), "units: {:?}", units);
}
