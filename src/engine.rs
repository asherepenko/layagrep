//! Evaluator: builds states/questions within laya's token budget, consults the
//! cache first, and falls through to a judge backend (native or Python).
//!
//! Budget: laya's `max_len` is 512 tokens per question row and the question
//! head (instructions + options) takes up to 192. States are therefore capped
//! at [`STATE_CHAR_BUDGET`] characters (~300 tokens), leaving headroom so the
//! prefix — which always holds the query — survives `build_sequence`'s
//! right-side truncation.

use crate::cache::{Cache, Namespace};
use crate::judge::{self, Answers, Judge};
use crate::worker::Question;
use std::collections::HashMap;

pub const STATE_CHAR_BUDGET: usize = 1000;
pub const PROMPT_VERSION: &str = "layagrep-4";

/// Judges are expensive to create (process spawn + model load). The factory
/// defers that cost until the first cache miss, so fully-cached reruns never
/// pay it.
pub type JudgeFactory = Box<dyn FnMut() -> Result<Box<dyn Judge>, crate::judge::JudgeError>>;

pub struct Engine {
    factory: JudgeFactory,
    judge: Option<Box<dyn Judge>>,
    cache: Cache,
    namespace: Namespace,
    pub requests: u64,
    pub state_truncations: u32,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for EngineError {}

/// One evaluation: a state string plus questions, cache-keyed as a unit.
#[derive(Clone)]
pub struct Evaluation<'a> {
    pub state: &'a str,
    pub questions: &'a [(String, Question)],
}

impl Engine {
    pub fn new(model_id: &str, factory: JudgeFactory, cache: Cache) -> Self {
        Engine {
            // Cache answers are checkpoint outputs; backends agree to the 4th
            // decimal, so the namespace is backend-independent and cached
            // answers transfer across engines.
            namespace: Namespace {
                model: model_id.to_string(),
                dtype: String::new(),
                protocol: "laya-jsonl-1".to_string(),
                prompt_version: PROMPT_VERSION.to_string(),
            },
            factory,
            judge: None,
            cache,
            requests: 0,
            state_truncations: 0,
        }
    }

    /// Force judge creation (doctor path).
    pub fn spawn_now(&mut self) -> Result<(), EngineError> {
        self.ensure_judge().map(|_| ())
    }

    fn ensure_judge(&mut self) -> Result<&mut Box<dyn Judge>, EngineError> {
        if self.judge.is_none() {
            let judge = (self.factory)().map_err(|e| EngineError(e.0))?;
            self.judge = Some(judge);
        }
        Ok(self.judge.as_mut().expect("judge was just set"))
    }

    /// The live backend, if one was spawned.
    pub fn backend(&self) -> Option<&dyn Judge> {
        self.judge.as_deref()
    }

    pub fn cache_issues(&self) -> Vec<(String, u32)> {
        self.cache.issues()
    }

    pub fn cache_hits(&self) -> u64 {
        self.cache.hits
    }

    fn request_json(evaluation: &Evaluation) -> serde_json::Value {
        serde_json::json!({
            "state": evaluation.state,
            "questions": evaluation.questions.iter().map(|(id, q)| {
                (id.clone(), question_json(q))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
        })
    }

    /// Truncate a state to the char budget on a char boundary.
    pub fn clamp_state(state: &str) -> (&str, bool) {
        if state.len() <= STATE_CHAR_BUDGET {
            return (state, false);
        }
        let mut end = STATE_CHAR_BUDGET;
        while end > 0 && !state.is_char_boundary(end) {
            end -= 1;
        }
        (&state[..end], true)
    }

    pub fn evaluate(&mut self, evaluation: &Evaluation) -> Result<Answers, EngineError> {
        let (state, truncated) = Self::clamp_state(evaluation.state);
        if truncated {
            self.state_truncations += 1;
        }
        let request = Self::request_json(&Evaluation {
            state,
            questions: evaluation.questions,
        });
        let key = Cache::key(&self.namespace, &request);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached);
        }
        let questions: HashMap<String, Question> = evaluation
            .questions
            .iter()
            .map(|(id, q)| (id.clone(), q.clone()))
            .collect();
        let answers = self
            .ensure_judge()?
            .predict(state, &questions)
            .map_err(|e| EngineError(e.0))?;
        self.requests += 1;
        for (id, _) in evaluation.questions.iter() {
            match answers.get(id) {
                Some(value) if value.is_finite() && (0.0..=1.0).contains(value) => {}
                _ => {
                    return Err(EngineError(format!(
                        "judge returned invalid answer for question {:?}",
                        id
                    )))
                }
            }
        }
        self.cache.put(&key, &answers);
        Ok(answers)
    }

    /// Batch evaluation: cache lookups first, misses fan out to the pool in
    /// parallel. Returns per-input results in order.
    pub fn evaluate_many(
        &mut self,
        evaluations: &[Evaluation],
    ) -> Vec<Result<Answers, EngineError>> {
        let clamped: Vec<(&str, bool)> =
            evaluations.iter().map(|e| Self::clamp_state(e.state)).collect();
        let requests: Vec<serde_json::Value> = evaluations
            .iter()
            .zip(clamped.iter())
            .map(|(evaluation, (state, truncated))| {
                let _ = truncated;
                Self::request_json(&Evaluation {
                    state,
                    questions: evaluation.questions,
                })
            })
            .collect();
        for ((_, truncated), _) in clamped.iter().zip(requests.iter()) {
            if *truncated {
                self.state_truncations += 1;
            }
        }
        let keys: Vec<String> = requests
            .iter()
            .map(|request| Cache::key(&self.namespace, request))
            .collect();
        let mut cached: Vec<Option<Answers>> = Vec::with_capacity(evaluations.len());
        for key in &keys {
            cached.push(self.cache.get(key));
        }
        let mut misses: Vec<judge::Job> = evaluations
            .iter()
            .zip(clamped.iter())
            .zip(cached.iter())
            .filter_map(|((evaluation, (state, _)), hit)| {
                if hit.is_some() {
                    None
                } else {
                    let questions: HashMap<String, Question> = evaluation
                        .questions
                        .iter()
                        .map(|(id, q)| (id.clone(), q.clone()))
                        .collect();
                    Some((state.to_string(), questions))
                }
            })
            .collect();
        if !misses.is_empty() {
            let results = self
                .judge
                .as_mut()
                .map(|judge| judge.predict_many(std::mem::take(&mut misses)));
            let mut results = match results {
                Some(result) => result,
                None => {
                    // Lazy spawn on first miss.
                    match (self.factory)().map_err(|e| EngineError(e.0)) {
                        Ok(mut judge) => {
                            let out = judge.predict_many(std::mem::take(&mut misses));
                            self.judge = Some(judge);
                            out
                        }
                        Err(error) => {
                            return evaluations
                                .iter()
                                .map(|_| Err(error.clone()))
                                .collect();
                        }
                    }
                }
            };
            self.requests += results.len() as u64;
            let mut result_iter = results.drain(..);
            for (index, hit) in cached.iter_mut().enumerate() {
                if hit.is_none() {
                    if let Some(result) = result_iter.next() {
                        match result {
                            Ok(answers) => {
                                let valid = evaluations[index]
                                    .questions
                                    .iter()
                                    .all(|(id, _)| {
                                        matches!(answers.get(id),
                                            Some(v) if v.is_finite() && (0.0..=1.0).contains(v))
                                    });
                                if valid {
                                    self.cache.put(&keys[index], &answers);
                                    *hit = Some(answers);
                                }
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }
        evaluations
            .iter()
            .zip(cached.into_iter())
            .map(|(evaluation, hit)| match hit {
                Some(mut answers) => {
                    let missing = evaluation
                        .questions
                        .iter()
                        .any(|(id, _)| !answers.contains_key(id));
                    if missing {
                        Err(EngineError("incomplete answers".into()))
                    } else {
                        Ok(answers)
                    }
                }
                None => Err(EngineError("judge failed for state".into())),
            })
            .collect()
    }

    /// Convenience: single boolean question over a state.
    pub fn judge_bool(&mut self, state: &str, instructions: &str) -> Result<f64, EngineError> {
        let questions = vec![(
            "q".to_string(),
            Question::Noul { instructions: instructions.to_string() },
        )];
        let evaluation = Evaluation { state, questions: &questions };
        Ok(self.evaluate(&evaluation)?.remove("q").unwrap_or(0.0))
    }
}

pub fn question_json(question: &Question) -> serde_json::Value {
    match question {
        Question::Noul { instructions } => serde_json::json!({
            "type": "noul",
            "instructions": instructions,
        }),
    }
}

/// Compact state prefix shared by every prompt: query first so it is
/// never the truncated side.
pub fn state_header(query: &str) -> String {
    format!("Query: {}", one_line(query))
}

pub fn one_line(text: &str) -> String {
    text.replace(['\n', '\r'], " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_clamp_respects_char_boundaries() {
        let state = "q".repeat(STATE_CHAR_BUDGET + 7);
        let (clamped, truncated) = Engine::clamp_state(&state);
        assert!(truncated);
        assert!(clamped.len() <= STATE_CHAR_BUDGET);
        let (same, truncated) = Engine::clamp_state("short");
        assert_eq!(same, "short");
        assert!(!truncated);
    }

    #[test]
    fn multibyte_boundary_is_safe() {
        let mut state = "квітка ".repeat(200);
        state.push_str("tail");
        let (clamped, _) = Engine::clamp_state(&state);
        assert!(clamped.len() <= STATE_CHAR_BUDGET);
    }
}
