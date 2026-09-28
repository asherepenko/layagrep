//! Pipeline contract test with a deterministic mock judge — no model needed.

#[path = "../src/cache.rs"]
mod cache;
#[path = "../src/engine.rs"]
mod engine;
#[path = "../src/judge.rs"]
mod judge;
#[path = "../src/rank.rs"]
mod rank;
#[path = "../src/render.rs"]
mod render;
#[path = "../src/retrieve.rs"]
mod retrieve;
#[path = "../src/selection.rs"]
mod selection;
#[path = "../src/source.rs"]
mod source;
#[path = "../src/types.rs"]
mod types;
#[path = "../src/walk.rs"]
mod walk;
#[path = "../src/worker.rs"]
mod worker;

use std::collections::HashMap;

/// Judge that answers from state content: files/units whose description
/// mentions "auth" are relevant, everything else is not.
struct AuthMock;

impl judge::Judge for AuthMock {
    fn predict(
        &mut self,
        state: &str,
        _questions: &HashMap<String, worker::Question>,
    ) -> Result<judge::Answers, judge::JudgeError> {
        let value = if state.contains("auth") { 0.95 } else { 0.05 };
        Ok(_questions
            .keys()
            .map(|id| (id.clone(), value))
            .collect())
    }
    fn namespace(&self) -> String {
        "mock".to_string()
    }
    fn describe(&self) -> String {
        "mock".to_string()
    }
}

fn write_repo(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("src/auth")).unwrap();
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::write(
        root.join("src/auth/login.py"),
        "\"\"\"Session login.\"\"\"\nimport hashlib\n\n\ndef verify_password(user, password):\n    \"\"\"Check the password hash.\"\"\"\n    return hashlib.sha256(password.encode()).hexdigest() == user.hash\n\n\nclass SessionManager:\n    def __init__(self, store):\n        self.store = store\n\n    def login(self, user, password):\n        if verify_password(user, password):\n            return self.store.create(user)\n        return None\n",
    )
    .unwrap();
    std::fs::write(
        root.join("docs/notes.md"),
        "# Notes\n\nUnrelated documentation about the build process.\n",
    )
    .unwrap();
    std::fs::write(root.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::create_dir_all(root.join("ignored")).unwrap();
    std::fs::write(root.join("ignored/secret.py"), "password = 'hunter2'\n").unwrap();
}

#[test]
fn retrieves_relevant_file_and_excerpts() {
    let root = std::env::temp_dir().join("layagrep-contract-test");
    let _ = std::fs::remove_dir_all(&root);
    write_repo(&root);

    let store = cache::Cache::new(root.join(".cache"), false);
    let mut factory: engine::JudgeFactory = Box::new(|| Ok(Box::new(AuthMock)));
    let mut engine = engine::Engine::new("mock-checkpoint", factory, store);
    let options = retrieve::SearchOptions {
        policy: walk::Policy::default(),
        query: "Where is the password verified during login?".to_string(),
        root: root.to_string_lossy().into_owned(),
        debug_scores: false,
    };
    let mut progress = retrieve::Progress::new(false);
    let result = retrieve::retrieve(&options, &mut engine, &mut progress, &|| false);

    assert_eq!(result.status, types::Status::Complete);
    let paths: Vec<&str> = result.files.iter().map(|f| f.path.as_str()).collect();
    assert!(paths.contains(&"src/auth/login.py"), "files: {:?}", paths);
    assert!(
        !paths.iter().any(|p| p.contains("ignored")),
        "gitignore must exclude ignored/: {:?}",
        paths
    );
    assert!(
        !paths.iter().any(|p| p.ends_with(".gitignore")),
        "hidden files excluded: {:?}",
        paths
    );

    let login = result.files.iter().find(|f| f.path == "src/auth/login.py").unwrap();
    assert!(
        login.excerpts.iter().any(|e| e.source.contains("verify_password")),
        "excerpt must contain the password check: {:?}",
        login.excerpts.iter().map(|e| e.source.clone()).collect::<Vec<_>>()
    );
    assert!(login.leads.iter().any(|l| l.name.contains("verify_password")));

    let rendered = render::render_result(&result, 0);
    assert!(rendered.starts_with("Layagrep:"), "{}", rendered.lines().next().unwrap());
    assert!(rendered.contains("Source block \"src/auth/login.py\""));
    assert!(rendered.trim_end().ends_with("End context."));
    let _ = std::fs::remove_dir_all(&root);
}
