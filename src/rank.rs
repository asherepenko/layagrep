//! Deterministic lexical ranking (BM25) over file descriptions, blended with
//! the laya judge's semantic grade. The judge's absolute file-level scores are
//! noise-heavy (measured: on-topic file 0.45, narrative video file 0.85), so
//! lexical grounding ranks and the judge gates: files with no query-term
//! overlap are dropped, files the judge firmly rejects (measured: LICENSE
//! 0.10, lockfile 0.15) cannot rank on stray terms alone.

use std::collections::HashMap;

const K1: f64 = 1.2;
const B: f64 = 0.75;

fn push_word(token: String, out: &mut Vec<String>) {
    if token.len() >= 2 {
        for variant in variants(&token.to_ascii_lowercase()) {
            out.push(variant);
        }
    }
}

/// Inflection variants so query and document terms intersect across
/// cache/cached/caches, request/requests, parser/parsing. Each token indexes
/// itself plus suffix-stripped forms; matching on any shared variant counts.
fn variants(word: &str) -> Vec<String> {
    let mut set: Vec<String> = vec![word.to_string()];
    for suffix in ["ing", "ies", "ed", "es", "er", "s", "d"] {
        if let Some(stripped) = word.strip_suffix(suffix) {
            if stripped.len() >= 3 {
                set.push(stripped.to_string());
            }
        }
    }
    let bases: Vec<String> = set.clone();
    for base in bases {
        if let Some(shorter) = base.strip_suffix('e') {
            if shorter.len() >= 3 {
                set.push(shorter.to_string());
            }
        }
    }
    set.sort();
    set.dedup();
    set
}

/// Split into words: non-alphanumeric boundaries, snake_case, camelCase,
/// kebab-case, and path separators.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for chunk in text.split(|c: char| !(c.is_ascii_alphanumeric())) {
        if chunk.is_empty() {
            continue;
        }
        // Split camelCase / PascalCase humps.
        let mut start = 0usize;
        let bytes = chunk.as_bytes();
        for index in 0..bytes.len() {
            let next = index + 1;
            if next < bytes.len() {
                let current_upper = bytes[index].is_ascii_uppercase();
                let next_upper = bytes[next].is_ascii_uppercase();
                let after_next_upper =
                    next + 1 < bytes.len() && bytes[next + 1].is_ascii_uppercase();
                if current_upper && next_upper && !after_next_upper && next - start > 1 {
                    push_word(chunk[start..next].to_string(), &mut tokens);
                    start = next;
                } else if !current_upper && next_upper {
                    push_word(chunk[start..next].to_string(), &mut tokens);
                    start = next;
                }
            }
        }
        push_word(chunk[start..].to_string(), &mut tokens);
    }
    tokens
}

pub struct Bm25Index {
    doc_terms: Vec<Vec<String>>,
    doc_freq: Vec<HashMap<String, usize>>,
    avg_len: f64,
    doc_count: f64,
    idf: HashMap<String, f64>,
}

impl Bm25Index {
    pub fn new(documents: &[String]) -> Self {
        let mut doc_terms = Vec::with_capacity(documents.len());
        let mut doc_freq = Vec::with_capacity(documents.len());
        let mut total_len = 0usize;
        let mut df: HashMap<String, usize> = HashMap::new();
        for document in documents {
            let terms = tokenize(document);
            total_len += terms.len();
            let mut counts: HashMap<String, usize> = HashMap::new();
            for term in &terms {
                *counts.entry(term.clone()).or_insert(0) += 1;
            }
            for term in counts.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doc_terms.push(terms);
            doc_freq.push(counts);
        }
        let doc_count = documents.len() as f64;
        let idf = df
            .into_iter()
            .map(|(term, frequency)| {
                let value = ((doc_count - frequency as f64 + 0.5)
                    / (frequency as f64 + 0.5)
                    + 1.0)
                    .ln();
                (term, value)
            })
            .collect();
        Bm25Index {
            doc_terms,
            doc_freq,
            avg_len: if documents.is_empty() { 1.0 } else { total_len as f64 / doc_count },
            doc_count,
            idf,
        }
    }

    /// BM25 score of one document (by index) against the query terms.
    pub fn score(&self, doc_index: usize, query: &[String]) -> f64 {
        let length = self.doc_terms[doc_index].len() as f64;
        let norm = K1 * (1.0 - B + B * length / self.avg_len);
        let mut total = 0.0;
        for term in query {
            let tf = self.doc_freq[doc_index].get(term).copied().unwrap_or(0) as f64;
            if tf == 0.0 {
                continue;
            }
            let idf = self.idf.get(term).copied().unwrap_or(0.0);
            total += idf * tf * (K1 + 1.0) / (tf + norm);
        }
        total
    }

    /// Scores for all documents, normalized to [0, 1] against the best.
    pub fn scores(&self, query: &[String]) -> Vec<f64> {
        let raw: Vec<f64> = (0..self.doc_terms.len())
            .map(|index| self.score(index, query))
            .collect();
        let max = raw.iter().cloned().fold(0.0_f64, f64::max);
        if max <= 0.0 {
            return raw;
        }
        raw.into_iter().map(|value| value / max).collect()
    }
}

/// A search corpus: documents keyed by file path.
pub struct DescriptionCorpus {
    pub paths: Vec<String>,
    index: Bm25Index,
}

impl DescriptionCorpus {
    pub fn new(paths: Vec<String>, documents: Vec<String>) -> Self {
        DescriptionCorpus { paths, index: Bm25Index::new(&documents) }
    }

    /// Normalized lexical scores aligned with `paths`.
    pub fn rank(&self, query: &str) -> Vec<f64> {
        let terms = tokenize(query);
        self.index.scores(&terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_identifiers_and_paths() {
        let tokens = tokenize("packages/core/src/createEvaluationCache.ts rate-limited");
        assert!(tokens.contains(&"packages".to_string()));
        assert!(tokens.contains(&"create".to_string()));
        assert!(tokens.contains(&"evaluation".to_string()));
        assert!(tokens.contains(&"cache".to_string()));
        assert!(tokens.contains(&"rate".to_string()));
        assert!(tokens.contains(&"limited".to_string()));
    }

    #[test]
    fn variants_intersect_across_inflections() {
        let cached = variants("cached");
        let cache = variants("cache");
        assert!(cached.iter().any(|v| cache.contains(v)), "cached ∩ cache");
        let requests = variants("requests");
        let request = variants("request");
        assert!(requests.iter().any(|v| request.contains(v)));
        let parsing = variants("parsing");
        let parser = variants("parser");
        assert!(parsing.iter().any(|v| parser.contains(v)));
    }

    #[test]
    fn bm25_ranks_matching_document_first() {
        let docs = vec![
            "File: video/stations/Cut.tsx Declarations: BLADE, SW, slotX, BIN, tumble".to_string(),
            "File: packages/core/src/cache.ts Declarations: createEvaluationCache, CacheInput, put, get, trim"
                .to_string(),
            "File: scripts/licenses/bzip2.LICENSE Content: redistribution".to_string(),
        ];
        let corpus = DescriptionCorpus::new(
            vec!["a".into(), "b".into(), "c".into()],
            docs,
        );
        let scores = corpus.rank("evaluation requests cached rate-limited");
        assert!(scores[1] > scores[0], "cache.ts must outrank Cut.tsx");
        assert!(scores[1] > scores[2]);
        assert!(scores[2] == 0.0, "no-overlap document must score zero");
    }
}
