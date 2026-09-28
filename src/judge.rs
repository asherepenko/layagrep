//! Judge backends: interchangeable evaluation engines behind one trait.
//!
//! - [`NativeJudge`] runs laya-rs (candle) in-process — single binary, no
//!   Python, ~50–100 ms per question on CPU with Accelerate.
//! - [`PythonJudge`] drives the installed laya-mlx package through a resident
//!   worker process — GPU speed on Apple Silicon, the reference port.
//!
//! `auto` prefers Python when `laya_mlx` is importable, else native.

use crate::worker::{self, Question, Worker};
use std::collections::HashMap;
use std::time::Duration;

pub type Answers = HashMap<String, f64>;

#[derive(Debug)]
pub struct JudgeError(pub String);

impl std::fmt::Display for JudgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for JudgeError {}

pub trait Judge {
    fn predict(
        &mut self,
        state: &str,
        questions: &HashMap<String, Question>,
    ) -> Result<Answers, JudgeError>;
    /// Identity feeds the cache namespace: answers differ per backend/dtype.
    fn namespace(&self) -> String;
    fn describe(&self) -> String;
}

// ---------------------------------------------------------------------------
// Native (laya-rs)
// ---------------------------------------------------------------------------

pub const DEFAULT_MODEL_ID: &str = "aac6fef/laya-mlx";

pub struct NativeJudge {
    agent: laya::Agent,
    model_id: String,
}

impl NativeJudge {
    pub fn load(model_id: &str, dtype: &str) -> Result<Self, JudgeError> {
        use laya::Agent;
        // CPU + Accelerate has no f16 matmul; f32 is also the faster CPU path.
        let dtype = match dtype {
            "float16" | "bfloat16" => "float32",
            other => other,
        };
        let dtype = laya::agent::parse_dtype(dtype).map_err(|e| JudgeError(e.to_string()))?;
        let start = std::time::Instant::now();
        let builder = Agent::builder().dtype(dtype);
        let agent = if std::path::Path::new(model_id).exists() {
            builder
                .build(model_id)
                .map_err(|e| JudgeError(format!("native load {}: {}", model_id, e)))?
        } else {
            Agent::from_pretrained_with(model_id, None, None, builder)
                .map_err(|e| JudgeError(format!("native load {}: {}", model_id, e)))?
        };
        eprintln!(
            "[layagrep] native engine ready ({}) in {:.2}s",
            model_id,
            start.elapsed().as_secs_f64()
        );
        Ok(NativeJudge { agent, model_id: model_id.to_string() })
    }
}

impl Judge for NativeJudge {
    fn predict(
        &mut self,
        state: &str,
        questions: &HashMap<String, Question>,
    ) -> Result<Answers, JudgeError> {
        let state_value = serde_json::Value::String(state.to_string());
        let mut question_map = serde_json::Map::new();
        for (id, question) in questions {
            question_map.insert(
                id.clone(),
                match question {
                    Question::Noul { instructions } => serde_json::json!({
                        "type": "noul",
                        "instructions": instructions,
                    }),
                },
            );
        }
        let questions_value = serde_json::Value::Object(question_map);
        let prediction = self
            .agent
            .predict(&state_value, &questions_value)
            .map_err(|e| JudgeError(format!("native predict: {}", e)))?;
        let mut answers = Answers::new();
        for (id, _question) in questions {
            if let Some(answer) = prediction.answer(id) {
                // noul: P(true) is option index 1.
                answers.insert(id.clone(), answer.probabilities.get(1).copied().unwrap_or(0.0));
            }
        }
        Ok(answers)
    }

    fn namespace(&self) -> String {
        format!("native-cpu-{}", self.model_id)
    }

    fn describe(&self) -> String {
        format!(
            "laya-rs native ({}), device {:?} (f16 promoted to f32 on CPU)",
            self.model_id,
            self.agent.device()
        )
    }
}

// ---------------------------------------------------------------------------
// Python worker (laya-mlx)
// ---------------------------------------------------------------------------

pub struct PythonJudge {
    worker: Worker,
    model_id: String,
    dtype: String,
}

impl PythonJudge {
    pub fn launch(
        model_id: &str,
        dtype: &str,
        python: Option<&str>,
        ready_timeout: Duration,
    ) -> Result<Self, JudgeError> {
        let worker = Worker::spawn(model_id, dtype, python, ready_timeout)
            .map_err(|e| JudgeError(format!("python worker: {}", e)))?;
        Ok(PythonJudge {
            worker,
            model_id: model_id.to_string(),
            dtype: dtype.to_string(),
        })
    }
}

impl Judge for PythonJudge {
    fn predict(
        &mut self,
        state: &str,
        questions: &HashMap<String, Question>,
    ) -> Result<Answers, JudgeError> {
        let answers = self
            .worker
            .predict(state, questions, Duration::from_secs(120))
            .map_err(|e| JudgeError(e.to_string()))?;
        Ok(answers)
    }

    fn namespace(&self) -> String {
        format!("python-mlx-{}-{}", self.model_id, self.dtype)
    }

    fn describe(&self) -> String {
        format!(
            "laya-mlx python worker ({} {}), interpreter {}",
            self.model_id,
            self.dtype,
            self.worker.python_origin()
        )
    }
}

/// Python availability probe without launching the full worker.
pub fn python_laya_available() -> bool {
    match worker::find_python(None) {
        Ok((python, _origin)) => {
            let probed = python.ends_with("python3") || python.ends_with("python");
            if probed {
                // The fallback path already probed importability.
                return true;
            }
            // Shebang interpreter: probe import directly.
            std::process::Command::new(python)
                .arg("-c")
                .arg("import laya_mlx")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        }
        Err(_) => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum EngineKind {
    Auto,
    Native,
    Python,
}

pub fn create_judge(
    kind: EngineKind,
    model_id: &str,
    dtype: &str,
    python: Option<&str>,
) -> Result<Box<dyn Judge>, JudgeError> {
    match kind {
        EngineKind::Native => Ok(Box::new(NativeJudge::load(model_id, dtype)?)),
        EngineKind::Python => Ok(Box::new(PythonJudge::launch(
            model_id,
            dtype,
            python,
            Duration::from_secs(600),
        )?)),
        EngineKind::Auto => {
            if python_laya_available() {
                if let Ok(judge) = PythonJudge::launch(model_id, dtype, python, Duration::from_secs(600)) {
                    return Ok(Box::new(judge));
                }
            }
            Ok(Box::new(NativeJudge::load(model_id, dtype)?))
        }
    }
}
