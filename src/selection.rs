//! Per-candidate source selection, batched across files for parallel judging:
//! build unit groups for every candidate, fan the evaluations out through the
//! pool, then assemble per-file outcomes (merge, expand, excerpt) on the CPU.

use crate::engine::{state_header, Engine};
use crate::source::{lines_text, Inspection, LineIndex, Range, SourceUnit};
use crate::types::{Excerpt, ReadingLead};
use crate::walk::Snapshot;

const GROUP_UNIT_LIMIT: usize = 2;
/// Units per file that reach the judge: ranked lexically first, so files with
/// dozens of declarations do not multiply cold-search requests.
const MAX_JUDGED_UNITS: usize = 14;
const GROUP_SOURCE_BUDGET: usize = 380;
const WINDOW_LINES: u32 = 3;
const LEAD_THRESHOLD: f64 = 0.25;
const SELECT_THRESHOLD: f64 = 0.5;
const MAX_EXCERPT_BYTES: usize = 24_000;
const BATCH_CHUNK: usize = 96;

pub struct SelectionOutcome {
    pub selected: Vec<Range>,
    pub rendered: Vec<Range>,
    pub excerpts: Vec<Excerpt>,
    pub leads: Vec<ReadingLead>,
}

pub struct SelectionInput<'a> {
    pub snapshot: &'a Snapshot,
    pub inspection: Option<&'a Inspection>,
}

fn unit_instructions(name: &str, range: Range) -> String {
    format!(
        "Does the declaration {} (lines {}-{}) implement, control, or test the behavior the query asks about? \
Count the current implementation even if buggy; generic terminology and unrelated utilities do not count.",
        name, range.start_line, range.end_line
    )
}

fn signature_line(source: &str, index: &LineIndex, range: Range) -> String {
    let mut text = lines_text(
        source,
        index,
        Range { start_line: range.start_line, end_line: range.start_line },
    );
    if text.trim().is_empty() && range.end_line > range.start_line {
        text = lines_text(
            source,
            index,
            Range { start_line: range.start_line + 1, end_line: range.start_line + 1 },
        );
    }
    text.trim().chars().take(70).collect()
}

fn unit_docstring(source: &str, index: &LineIndex, range: Range) -> Option<String> {
    for line_number in range.start_line..(range.start_line + 6).min(range.end_line + 1) {
        let line = lines_text(
            source,
            index,
            Range { start_line: line_number, end_line: line_number },
        );
        let trimmed = line.trim();
        if trimmed.starts_with("///")
            || trimmed.starts_with("/**")
            || trimmed.starts_with('#')
            || trimmed.starts_with("\"\"\"")
        {
            let text = trimmed
                .trim_matches(|c: char| c == '/' || c == '*' || c == '"' || c == '#')
                .trim();
            if text.chars().any(|c| c.is_alphabetic()) {
                return Some(text.chars().take(90).collect());
            }
        }
    }
    None
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
    let mut calls = Vec::new();
    for (name, def_range) in defs {
        if def_range.start_line >= unit.range.start_line
            && def_range.end_line <= unit.range.end_line
        {
            continue; // nested/own definition
        }
        for line_number in unit.range.start_line..=unit.range.end_line {
            let line = lines_text(
                &snapshot.source,
                index,
                Range { start_line: line_number, end_line: line_number },
            );
            if line.contains(&format!("{}(", name)) {
                calls.push((*name).to_string());
                break;
            }
        }
    }
    calls.truncate(4);
    if !calls.is_empty() {
        parts.push(format!("calls {}", calls.join(", ")));
    }
    parts.join(" | ")
}

struct UnitGroup {
    unit_indices: Vec<usize>,
    state: String,
    questions: Vec<(String, crate::worker::Question)>,
}

fn build_groups(snapshot: &Snapshot, inspection: &Inspection, query: &str) -> Vec<UnitGroup> {
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
    if inspection.units.len() <= MAX_JUDGED_UNITS {
        return build_groups_for(&index, inspection, &(0..inspection.units.len()).collect::<Vec<_>>(), snapshot, query, &defs);
    }
    // Rank units lexically against the query; judge only the strongest.
    let descriptions: Vec<String> = inspection
        .units
        .iter()
        .map(|unit| describe_unit(snapshot, &index, unit, &defs))
        .collect();
    let corpus = crate::rank::DescriptionCorpus::new(
        (0..inspection.units.len()).map(|i| i.to_string()).collect(),
        descriptions.clone(),
    );
    let ranked = corpus.rank(query);
    let mut order: Vec<usize> = (0..inspection.units.len()).collect();
    order.sort_by(|a, b| {
        ranked[*b]
            .partial_cmp(&ranked[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let chosen: Vec<usize> = order.into_iter().take(MAX_JUDGED_UNITS).collect();
    build_groups_for(&index, inspection, &chosen, snapshot, query, &defs)
}

fn build_groups_for(
    index: &LineIndex,
    inspection: &Inspection,
    chosen: &[usize],
    snapshot: &Snapshot,
    query: &str,
    defs: &[(&str, &Range)],
) -> Vec<UnitGroup> {
    let mut groups: Vec<UnitGroup> = Vec::new();
    let mut group_bytes = 0usize;
    let mut current: Vec<usize> = Vec::new();
    for unit_index in chosen.iter().copied() {
        let unit = &inspection.units[unit_index];
        let cost = unit.name.len() + 140;
        if !current.is_empty()
            && (current.len() >= GROUP_UNIT_LIMIT || group_bytes + cost > GROUP_SOURCE_BUDGET)
        {
            let drained = std::mem::take(&mut current);
            groups.push(finish_group(snapshot, &index, inspection, &drained, query, &defs));
            group_bytes = 0;
        }
        current.push(unit_index);
        group_bytes += cost;
    }
    if !current.is_empty() {
        groups.push(finish_group(snapshot, &index, inspection, &current, query, &defs));
    }
    groups
}

fn finish_group(
    snapshot: &Snapshot,
    index: &LineIndex,
    inspection: &Inspection,
    unit_indices: &[usize],
    query: &str,
    defs: &[(&str, &Range)],
) -> UnitGroup {
    let mut state = format!("{}\nFile: {}\nDeclarations:\n", state_header(query), snapshot.path);
    let mut questions = Vec::new();
    for (position, unit_index) in unit_indices.iter().enumerate() {
        let unit = &inspection.units[*unit_index];
        state.push_str(&describe_unit(snapshot, index, unit, defs));
        state.push('\n');
        questions.push((
            format!("q{}", position),
            crate::worker::Question::Noul {
                instructions: unit_instructions(&unit.name, unit.range),
            },
        ));
    }
    UnitGroup { unit_indices: unit_indices.to_vec(), state, questions }
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

fn assemble_outcome(
    snapshot: &Snapshot,
    inspection: &Inspection,
    unit_scores: &[f64],
) -> SelectionOutcome {
    let index = LineIndex::new(&snapshot.source);
    let line_count = index.line_count as u32;
    let mut selected: Vec<Range> = Vec::new();
    let mut leads: Vec<ReadingLead> = Vec::new();
    for (unit, score) in inspection.units.iter().zip(unit_scores.iter()) {
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

    SelectionOutcome {
        selected,
        rendered,
        excerpts,
        leads: deduped,
    }
}

/// Batch selection across files: unit groups from every file are judged in
/// parallel through the engine pool, then outcomes assemble per file. Returns
/// one `(outcome, issues)` pair per input, in input order.
pub fn select_files(
    inputs: &[SelectionInput],
    query: &str,
    engine: &mut Engine,
    progress: &mut dyn FnMut(usize),
) -> Vec<(SelectionOutcome, Vec<(&'static str, u32)>)> {
    // Resolve inspections once; derive missing ones.
    let owned: Vec<Inspection> = inputs
        .iter()
        .map(|input| match input.inspection {
            Some(inspection) => inspection.clone(),
            None => crate::source::inspect(input.snapshot),
        })
        .collect();

    // Build unit groups per file.
    let per_file: Vec<Vec<UnitGroup>> = inputs
        .iter()
        .zip(owned.iter())
        .map(|(input, inspection)| build_groups(input.snapshot, inspection, query))
        .collect();

    // Flatten groups across files in submission order.
    let mut layout: Vec<(usize, usize)> = Vec::new(); // (file index, group index)
    let mut jobs: Vec<(String, Vec<(String, crate::worker::Question)>)> = Vec::new();
    for (file_index, groups) in per_file.iter().enumerate() {
        for (group_index, group) in groups.iter().enumerate() {
            jobs.push((group.state.clone(), group.questions.clone()));
            layout.push((file_index, group_index));
        }
    }

    // Judge in chunks; answers arrive in submission order.
    let mut all_answers: Vec<Option<Vec<f64>>> = Vec::with_capacity(jobs.len());
    for chunk in jobs.chunks(BATCH_CHUNK) {
        let evaluations: Vec<crate::engine::Evaluation> = chunk
            .iter()
            .map(|(state, questions)| crate::engine::Evaluation { state, questions })
            .collect();
        let results = engine.evaluate_many(&evaluations);
        for result in results {
            match result {
                Ok(answers) => {
                    let count = answers.len();
                    let scores: Vec<f64> = (0..count)
                        .map(|i| *answers.get(&format!("q{}", i)).unwrap_or(&0.0))
                        .collect();
                    all_answers.push(Some(scores));
                }
                Err(_) => all_answers.push(None),
            }
        }
        progress(chunk.len());
    }

    // Scatter group answers back to their files.
    let mut group_answers: Vec<Vec<Option<Vec<f64>>>> =
        per_file.iter().map(|groups| vec![None; groups.len()]).collect();
    for ((file_index, group_index), answer) in layout.iter().zip(all_answers.into_iter()) {
        group_answers[*file_index][*group_index] = answer;
    }

    // Assemble per-file outcomes.
    let mut outcomes = Vec::with_capacity(inputs.len());
    for (((input, inspection), groups), answers) in inputs
        .iter()
        .zip(owned.iter())
        .zip(per_file.iter())
        .zip(group_answers.into_iter())
    {
        let mut unit_scores = vec![0.0_f64; inspection.units.len()];
        let mut provider_failures = 0u32;
        for (group, answer) in groups.iter().zip(answers.into_iter()) {
            match answer {
                Some(scores) => {
                    for (unit_index, score) in group.unit_indices.iter().zip(scores.iter()) {
                        if *unit_index < unit_scores.len() {
                            unit_scores[*unit_index] = *score;
                        }
                    }
                }
                None => provider_failures += 1,
            }
        }
        let issues = if provider_failures > 0 {
            vec![("provider", provider_failures)]
        } else {
            Vec::new()
        };
        outcomes.push((assemble_outcome(input.snapshot, inspection, &unit_scores), issues));
    }
    outcomes
}
