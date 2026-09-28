//! Per-candidate source selection: judge declaration units by description
//! (name, signature, local calls, docstring — calibrated: 0.56–0.72 relevant
//! vs 0.09–0.12 irrelevant), merge selected spans, expand windows with
//! adjacent comments, and build verbatim excerpts.

use crate::engine::{state_header, Engine};
use crate::source::{lines_text, Inspection, LineIndex, Range, SourceUnit};
use crate::types::{Excerpt, ReadingLead};
use crate::walk::Snapshot;

const GROUP_UNIT_LIMIT: usize = 2;
const GROUP_SOURCE_BUDGET: usize = 380;
const WINDOW_LINES: u32 = 3;
const LEAD_THRESHOLD: f64 = 0.25;
const SELECT_THRESHOLD: f64 = 0.5;
const MAX_EXCERPT_BYTES: usize = 24_000;

pub struct SelectionOutcome {
    pub selected: Vec<Range>,
    pub rendered: Vec<Range>,
    pub excerpts: Vec<Excerpt>,
    pub leads: Vec<ReadingLead>,
}

fn unit_instructions(name: &str, range: Range, query: &str) -> String {
    let _ = query; // the query lives in the state header; echoing it per question doubles head tokens
    format!(
        "Does the declaration {} (lines {}-{}) implement, control, or test the behavior the query asks about? \
Count the current implementation even if buggy; generic terminology and unrelated utilities do not count.",
        name, range.start_line, range.end_line
    )
}

fn signature_line(source: &str, index: &LineIndex, range: Range) -> String {
    let mut text = lines_text(source, index, Range { start_line: range.start_line, end_line: range.start_line });
    if text.trim().is_empty() && range.end_line > range.start_line {
        text = lines_text(source, index, Range { start_line: range.start_line + 1, end_line: range.start_line + 1 });
    }
    text.trim().chars().take(70).collect()
}

fn unit_docstring(source: &str, index: &LineIndex, range: Range) -> Option<String> {
    for line_number in range.start_line..(range.start_line + 6).min(range.end_line + 1) {
        let line = lines_text(source, index, Range { start_line: line_number, end_line: line_number });
        let trimmed = line.trim();
        if trimmed.starts_with("///") || trimmed.starts_with("/**") || trimmed.starts_with('#')
            || trimmed.starts_with("\"\"\"")
        {
            let text = trimmed.trim_matches(|c: char| c == '/' || c == '*' || c == '"' || c == '#').trim();
            if text.chars().any(|c| c.is_alphabetic()) {
                return Some(text.chars().take(90).collect());
            }
        }
    }
    None
}

/// Names of local definitions called inside the unit's body.
fn local_calls(
    source: &str,
    index: &LineIndex,
    unit: &SourceUnit,
    defs: &[(&str, &Range)],
) -> Vec<String> {
    let mut calls = Vec::new();
    for (name, def_range) in defs {
        if def_range.start_line >= unit.range.start_line
            && def_range.end_line <= unit.range.end_line
        {
            continue; // nested/own definition
        }
        for line_number in unit.range.start_line..=unit.range.end_line {
            let line = lines_text(source, index, Range { start_line: line_number, end_line: line_number });
            if line.contains(&format!("{}(", name)) {
                calls.push((*name).to_string());
                break;
            }
        }
    }
    calls.truncate(4);
    calls
}

fn describe_unit(
    snapshot: &Snapshot,
    index: &LineIndex,
    unit: &SourceUnit,
    defs: &[(&str, &Range)],
) -> String {
    let mut parts = vec![format!(
        "{} (lines {}-{}): {}",
        unit.name,
        unit.range.start_line,
        unit.range.end_line,
        signature_line(&snapshot.source, index, unit.range)
    )];
    if parts[0].len() < 60 {
        if let Some(doc) = unit_docstring(&snapshot.source, index, unit.range) {
            parts.push(doc);
        }
    }
    let calls = local_calls(&snapshot.source, index, unit, defs);
    if !calls.is_empty() {
        parts.push(format!("calls {}", calls.join(", ")));
    }
    parts.join(" | ")
}

/// Judge every unit; returns per-unit scores in unit order.
fn judge_units(
    snapshot: &Snapshot,
    inspection: &Inspection,
    query: &str,
    engine: &mut Engine,
    provider_failures: &mut u32,
) -> Vec<f64> {
    let index = LineIndex::new(&snapshot.source);
    let defs: Vec<(&str, &Range)> = inspection
        .units
        .iter()
        .filter(|unit| !unit.name.ends_with(".context") && unit.name != "source")
        .map(|unit| {
            let bare = unit.name.rsplit('.').next().unwrap_or(&unit.name);
            (bare, &unit.range)
        })
        .collect();
    let mut scores = Vec::with_capacity(inspection.units.len());
    let mut group: Vec<&SourceUnit> = Vec::new();
    let mut group_bytes = 0usize;

    let mut flush = |group: &mut Vec<&SourceUnit>, engine: &mut Engine| {
        if group.is_empty() {
            return;
        }
        let mut state = format!("{}\nFile: {}\nDeclarations:\n", state_header(query), snapshot.path);
        for unit in group.iter() {
            state.push_str(&describe_unit(snapshot, &index, unit, &defs));
            state.push('\n');
        }
        let questions: Vec<(String, crate::worker::Question)> = group
            .iter()
            .enumerate()
            .map(|(i, unit)| {
                (
                    format!("q{}", i),
                    crate::worker::Question::Noul {
                        instructions: unit_instructions(&unit.name, unit.range, query),
                    },
                )
            })
            .collect();
        match engine.evaluate(&crate::engine::Evaluation {
            state: &state,
            questions: &questions,
        }) {
            Ok(answers) => {
                for (i, _) in group.iter().enumerate() {
                    scores.push(answers.get(&format!("q{}", i)).copied().unwrap_or(0.0));
                }
            }
            Err(_) => {
                *provider_failures += 1;
                scores.extend(std::iter::repeat(0.0).take(group.len()));
            }
        }
        group.clear();
    };

    for unit in &inspection.units {
        let cost = unit.name.len() + 140; // descriptions are bounded by construction
        if !group.is_empty()
            && (group.len() >= GROUP_UNIT_LIMIT || group_bytes + cost > GROUP_SOURCE_BUDGET)
        {
            flush(&mut group, engine);
            group_bytes = 0;
        }
        group.push(unit);
        group_bytes += cost;
    }
    flush(&mut group, engine);
    scores
}

fn merge_ranges(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.retain(|r| r.end_line >= r.start_line);
    ranges.sort_by_key(|r| (r.start_line, r.end_line));
    let mut merged: Vec<Range> = Vec::new();
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start_line <= last.end_line.saturating_add(1) => {
                last.end_line = last.end_line.max(range.end_line);
            }
            _ => merged.push(range),
        }
    }
    merged
}

/// Expand a range by ±`window` lines and absorb comments separated only by
/// blank lines, mirroring jevgrep's excerpt windows.
fn expand_with_comments(
    range: Range,
    index: &LineIndex,
    line_count: u32,
    comments: &[Range],
) -> Range {
    let mut window = Range {
        start_line: range.start_line.saturating_sub(WINDOW_LINES).max(1),
        end_line: (range.end_line + WINDOW_LINES).min(line_count),
    };
    let blank_between = |a_end: u32, b_start: u32| -> bool {
        if a_end + 1 >= b_start {
            return true; // overlap or adjacent
        }
        (a_end + 1..b_start).all(|line| {
            let start = index.line_start(line);
            let end = index.line_end(line);
            start >= end
        })
    };
    let mut changed = true;
    while changed {
        changed = false;
        for comment in comments {
            let touching = (comment.start_line <= window.end_line
                && blank_between(comment.start_line, window.end_line))
                || (comment.end_line <= window.start_line
                    && blank_between(window.start_line, comment.end_line));
            if touching {
                let start = window.start_line.min(comment.start_line);
                let end = window.end_line.max(comment.end_line);
                if start != window.start_line || end != window.end_line {
                    window = Range { start_line: start, end_line: end };
                    changed = true;
                }
            }
        }
    }
    window
}

pub fn select_file(
    snapshot: &Snapshot,
    inspection: Option<&Inspection>,
    query: &str,
    engine: &mut Engine,
) -> (SelectionOutcome, Vec<(&'static str, u32)>) {
    let debug = std::env::var_os("LAYAGREP_DEBUG_UNITS").is_some();
    let mut issues: Vec<(&'static str, u32)> = Vec::new();
    let index = LineIndex::new(&snapshot.source);
    let line_count = index.line_count as u32;

    let fallback_inspection;
    let inspection = match inspection {
        Some(inspection) => inspection,
        None => {
            fallback_inspection = crate::source::inspect(snapshot);
            &fallback_inspection
        }
    };

    let mut provider_failures = 0u32;
    let judged = judge_units(snapshot, inspection, query, engine, &mut provider_failures);
    if debug {
        for (unit, score) in inspection.units.iter().zip(judged.iter()) {
            eprintln!("[unit] {:.3} {} {}:{}-{}", score, unit.name, snapshot.path, unit.range.start_line, unit.range.end_line);
        }
    }
    if provider_failures > 0 {
        issues.push(("provider", provider_failures));
    }

    let mut selected: Vec<Range> = Vec::new();
    let mut leads: Vec<ReadingLead> = Vec::new();
    for (unit, score) in inspection.units.iter().zip(judged.iter()) {
        if *score > SELECT_THRESHOLD {
            selected.push(unit.range);
        }
        if *score > LEAD_THRESHOLD {
            leads.push(ReadingLead { name: unit.name.clone(), range: unit.range, score: *score });
        }
    }
    let mut deduped: Vec<ReadingLead> = Vec::new();
    for lead in leads {
        match deduped.iter_mut().find(|l| l.name == lead.name && l.range == lead.range) {
            Some(existing) => existing.score = existing.score.max(lead.score),
            None => deduped.push(lead),
        }
    }
    deduped.sort_by(|a, b| a.range.start_line.cmp(&b.range.start_line).then(a.name.cmp(&b.name)));

    let selected = merge_ranges(selected);
    let windows: Vec<Range> = selected
        .iter()
        .map(|&range| expand_with_comments(range, &index, line_count, &inspection.comments))
        .collect();
    let rendered = merge_ranges(windows);
    let mut excerpts = Vec::new();
    for window in &rendered {
        let text = lines_text(&snapshot.source, &index, *window);
        let mut range = *window;
        if text.len() > MAX_EXCERPT_BYTES {
            let mut end = range.start_line;
            while end < range.end_line {
                let candidate = end + 1;
                if lines_text(
                    &snapshot.source,
                    &index,
                    Range { start_line: range.start_line, end_line: candidate },
                )
                .len()
                    > MAX_EXCERPT_BYTES
                {
                    break;
                }
                end = candidate;
            }
            range.end_line = end;
            excerpts.push(Excerpt {
                range,
                source: lines_text(&snapshot.source, &index, range),
            });
        } else {
            excerpts.push(Excerpt { range: *window, source: text });
        }
    }

    (
        SelectionOutcome {
            selected,
            rendered,
            excerpts,
            leads: deduped,
        },
        issues,
    )
}
