//! Embedded Python worker process: keeps the laya-mlx model resident.
//!
//! The worker script is embedded via `include_str!`, written once into the
//! layagrep cache directory, and executed by the interpreter that provides
//! `laya_mlx` (discovered from the `laya-mlx` launcher on PATH by default).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const WORKER_SCRIPT: &str = include_str!("worker.py");

#[derive(Debug)]
pub struct WorkerError(pub String);

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for WorkerError {}

/// A laya question definition sent to the worker.
#[derive(Debug, Clone)]
pub enum Question {
    /// P(true) for the instruction — the boolean workhorse.
    Noul { instructions: String },
}

impl Question {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Question::Noul { instructions } => serde_json::json!({
                "type": "noul",
                "instructions": instructions,
            }),
        }
    }
}

pub struct Worker {
    child: Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    next_id: u64,
    ready_info: serde_json::Value,
    pub load_seconds: f64,
    python: String,
}

fn home_cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            home.join(".cache")
        });
    base.join("layagrep")
}

/// Candidate interpreters, best first. The `laya-mlx` launcher's interpreter is
/// preferred because the tool guarantees `laya_mlx` is importable there.
pub fn find_python(explicit: Option<&str>) -> Result<(String, String), WorkerError> {
    if let Some(path) = explicit {
        return Ok((path.to_string(), "LAYAGREP_PYTHON".to_string()));
    }
    if let Ok(path) = std::env::var("LAYAGREP_PYTHON") {
        if !path.trim().is_empty() {
            return Ok((path, "LAYAGREP_PYTHON".to_string()));
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let launcher = dir.join("laya-mlx");
            if let Ok(mut file) = std::fs::File::open(&launcher) {
                let mut first = String::new();
                if std::io::BufReader::new(&mut file).read_line(&mut first).is_ok() {
                    let trimmed = first.trim_end();
                    if let Some(interp) = trimmed.strip_prefix("#!") {
                        let interp = interp.trim();
                        if !interp.is_empty() {
                            return Ok((
                                interp.to_string(),
                                format!("shebang of {}", launcher.display()),
                            ));
                        }
                    }
                }
            }
        }
    }
    // Fall back to python3 after probing that laya_mlx imports.
    let probe = Command::new("python3")
        .arg("-c")
        .arg("import laya_mlx")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Ok(("python3".to_string(), "python3 (laya_mlx import probe)".to_string()));
    }
    Err(WorkerError(
        "could not find a Python interpreter with laya_mlx installed; install laya-mlx \
         (pip install laya-mlx / uv tool install laya-mlx) or set LAYAGREP_PYTHON"
            .to_string(),
    ))
}

pub fn cache_dir() -> PathBuf {
    home_cache_dir()
}

fn worker_script_path() -> PathBuf {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(WORKER_SCRIPT.as_bytes());
    home_cache_dir().join(format!("worker-{}.py", hex::encode(&digest[..8])))
}

impl Worker {
    /// Spawn the worker and wait for its ready line.
    pub fn spawn(
        model: &str,
        dtype: &str,
        python: Option<&str>,
        ready_timeout: Duration,
    ) -> Result<Self, WorkerError> {
        let (python, origin) = find_python(python)?;
        let script = worker_script_path();
        if let Some(parent) = script.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| WorkerError(format!("cannot create cache dir {}: {}", parent.display(), e)))?;
        }
        std::fs::write(&script, WORKER_SCRIPT)
            .map_err(|e| WorkerError(format!("cannot write {}: {}", script.display(), e)))?;

        let started = Instant::now();
        let mut child = Command::new(&python)
            .arg("-u")
            .arg(&script)
            .arg("--model")
            .arg(model)
            .arg("--dtype")
            .arg(dtype)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| WorkerError(format!("cannot start python ({}): {}", python, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError("worker stdin unavailable".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError("worker stdout unavailable".to_string()))?;

        // Read the ready line on a thread so startup can time out.
        let (tx, rx) = mpsc::channel();
        let reader_thread = std::io::BufReader::new(stdout);
        std::thread::spawn(move || {
            let mut reader = reader_thread;
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => { let _ = tx.send(Err("worker exited before becoming ready".into())); return; }
                Ok(_) => {}
                Err(e) => { let _ = tx.send(Err(format!("worker stdout read failed: {}", e))); return; }
            }
            let _ = tx.send(Ok((line, reader)));
        });

        let (ready_line, reader) = rx
            .recv_timeout(ready_timeout)
            .map_err(|_| {
                let _ = child.kill();
                WorkerError(format!(
                    "laya worker did not become ready within {}s (model load or first download may be slow; retry)",
                    ready_timeout.as_secs()
                ))
            })?
            .map_err(WorkerError)?;

        let ready_info: serde_json::Value = serde_json::from_str(ready_line.trim())
            .map_err(|e| WorkerError(format!("invalid worker ready line: {} ({:?})", e, ready_line)))?;
        if ready_info.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            return Err(WorkerError(format!("worker not ready: {}", ready_line.trim())));
        }
        let load_seconds = started.elapsed().as_secs_f64();

        Ok(Worker {
            child,
            stdin,
            reader,
            next_id: 1,
            ready_info,
            load_seconds,
            python: format!("{} ({})", python, origin),
        })
    }

    pub fn python_origin(&self) -> &str {
        &self.python
    }

    pub fn model_info(&self) -> &serde_json::Value {
        &self.ready_info
    }

    /// One round trip: state + questions -> per-question numeric answers.
    pub fn predict(
        &mut self,
        state: &str,
        questions: &HashMap<String, Question>,
        timeout: Duration,
    ) -> Result<HashMap<String, f64>, WorkerError> {
        let id = self.next_id;
        self.next_id += 1;
        let request = serde_json::json!({
            "id": id,
            "state": state,
            "questions": questions.iter().map(|(qid, q)| (qid.clone(), q.to_json())).collect::<serde_json::Map<String, serde_json::Value>>(),
        });
        let line = serde_json::to_string(&request)
            .map_err(|e| WorkerError(format!("request encode failed: {}", e)))?;
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| WorkerError(format!("worker write failed: {}", e)))?;

        let started = Instant::now();
        let mut response = String::new();
        loop {
            if started.elapsed() > timeout {
                return Err(WorkerError(format!(
                    "worker response timed out after {}s",
                    timeout.as_secs_f64()
                )));
            }
            response.clear();
            let read = self
                .reader
                .read_line(&mut response)
                .map_err(|e| WorkerError(format!("worker read failed: {}", e)))?;
            if read == 0 {
                return Err(WorkerError("worker exited unexpectedly".to_string()));
            }
            let value: serde_json::Value = match serde_json::from_str(response.trim()) {
                Ok(v) => v,
                Err(_) => continue, // tolerate stray non-JSON output lines
            };
            if value.get("id").and_then(|v| v.as_u64()) != Some(id) {
                continue; // not ours (should not happen with sequential use)
            }
            if value.get("ok").and_then(|v| v.as_bool()) == Some(true) {
                let mut answers = HashMap::new();
                if let Some(map) = value.get("answers").and_then(|v| v.as_object()) {
                    for (qid, answer) in map {
                        if let Some(number) = answer.as_f64() {
                            answers.insert(qid.clone(), number);
                        }
                    }
                }
                return Ok(answers);
            }
            let message = value
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown worker error")
                .to_string();
            return Err(WorkerError(message));
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
