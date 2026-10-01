//! Output renderer, mirroring jevgrep's stdout contract: status header,
//! ranked file list, verbatim source blocks with line references, then
//! declaration and call locations, ending with `End context.`.

use crate::types::{FileEvidence, RetrievalResult, Status};

pub const DEFAULT_MAX_SOURCE_BYTES: usize = 0;

pub fn quote(value: &str) -> String {
    let escaped = serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
    escaped
        .chars()
        .filter(|c| {
            !matches!(c, '\u{7f}'..='\u{9f}' | '\u{2028}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        })
        .collect::<String>()
        .replace("\\u{7f}", "")
}

fn shell_arg(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Conservative test-file detection: test directory segments and conventional
/// test-file stems. False negatives merely keep a test in the source section;
/// false positives would demote real source, so edge shapes stay conservative.
pub fn is_test_path(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    for segment in lowered.split('/') {
        if matches!(segment, "test" | "tests" | "androidtest" | "testfixtures" | "__tests__") {
            return true;
        }
    }
    let file_name = lowered.rsplit('/').next().unwrap_or(lowered.as_str());
    let (stem, ext) = file_name.rsplit_once('.').unwrap_or((file_name, ""));
    match ext {
        "py" => stem.starts_with("test_") || stem.ends_with("_test"),
        "go" => stem.ends_with("_test"),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "mts" => {
            stem.ends_with(".test") || stem.ends_with(".spec") || stem.ends_with("_test")
        }
        _ => {
            // JVM/Swift CamelCase suffixes need the original casing:
            // "FooTest.kt" is a test, "latest.rs" is not.
            let original = path.rsplit('/').next().unwrap_or(path);
            let original_stem = original.rsplit_once('.').map(|(s, _)| s).unwrap_or(original);
            original_stem.ends_with("Test")
                || original_stem.ends_with("Tests")
                || original_stem.ends_with("IT")
                || stem.ends_with("_test")
        }
    }
}

/// Group and order files for presentation: source files (score desc) first,
/// then test files (score desc). Shared by the text report and the TUI so both
/// surfaces show the same structure.
pub fn grouped_files(result: &RetrievalResult) -> (Vec<&FileEvidence>, Vec<&FileEvidence>) {
    let mut main_files: Vec<&FileEvidence> = result
        .files
        .iter()
        .filter(|file| !is_test_path(&file.path))
        .collect();
    let mut test_files: Vec<&FileEvidence> = result
        .files
        .iter()
        .filter(|file| is_test_path(&file.path))
        .collect();
    let by_score = |a: &&FileEvidence, b: &&FileEvidence| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    };
    main_files.sort_by(by_score);
    test_files.sort_by(by_score);
    (main_files, test_files)
}

/// Present source before detailed reading leads so truncated output remains useful.
/// Test files group after source files and never dump excerpts: the judge may
/// rank them, but the agent's evidence budget belongs to production code.
pub fn render_result(result: &RetrievalResult, max_source_bytes: usize) -> String {
    let mut remaining = if max_source_bytes > 0 {
        max_source_bytes
    } else {
        usize::MAX
    };
    let mut files: Vec<(&FileEvidence, Vec<&crate::types::Excerpt>, bool, bool)> = Vec::new();
    let (main_files, test_files) = grouped_files(result);
    let ordered: Vec<&FileEvidence> = main_files.into_iter().chain(test_files).collect();
    for file in ordered {
        let is_test = is_test_path(&file.path);
        let mut omitted = file.source_omitted;
        let excerpts: Vec<&crate::types::Excerpt> = if is_test {
            Vec::new()
        } else {
            file.excerpts
                .iter()
                .filter(|excerpt| {
                    let bytes = excerpt.source.len();
                    if bytes > remaining {
                        omitted = true;
                        return false;
                    }
                    remaining -= bytes;
                    true
                })
                .collect()
        };
        files.push((file, excerpts, omitted, is_test));
    }

    let context = &result.repository_context;
    let omitted_count = files.iter().filter(|(_, _, omitted, _)| *omitted).count();
    let test_count = files.iter().filter(|(_, _, _, is_test)| *is_test).count();
    let with_source = files.iter().filter(|(_, excerpts, _, _)| !excerpts.is_empty()).count();
    let mut summary = format!("Layagrep: {} relevant files", files.len());
    if files.len() != with_source {
        summary.push_str(&format!(" · {} with source", with_source));
    }
    if test_count > 0 {
        summary.push_str(&format!(" · {} test files", test_count));
    }
    summary.push_str(if result.status != Status::Complete {
        "; discovery incomplete."
    } else {
        "."
    });
    let mut lines: Vec<String> = Vec::new();
    lines.push(summary);
    lines.push(
        "Symbols use name@start-end. Roles are estimates; locations-only files remain reading leads."
            .to_string(),
    );
    lines.push(format!(
        "AGENTS.md lookup (root and returned-file ancestors): {}{}.",
        if context.instruction_files.is_empty() {
            "none found".to_string()
        } else {
            context
                .instruction_files
                .iter()
                .map(|path| quote(path))
                .collect::<Vec<_>>()
                .join(", ")
        },
        if context.instruction_lookup_incomplete { "; lookup incomplete" } else { "" }
    ));
    if result.status == Status::Interrupted {
        lines.push("Interrupted.".to_string());
    }
    if omitted_count > 0 {
        lines.push(format!("Source omitted: {} file(s).", omitted_count));
    }
    for (kind, count) in &result.warnings {
        lines.push(format!("Warning: {}: {}", quote(kind), count));
    }
    for (kind, count) in &result.issues {
        lines.push(format!("Issue: {}: {}", quote(kind), count));
    }
    if let Some(failure) = &result.provider_failure {
        lines.push(format!("Provider error: {}", quote(failure)));
    }
    for path in &context.pytest_files {
        lines.push(format!(
            "Suggested test entry point (not executed): python -m pytest -q {}",
            shell_arg(path)
        ));
    }
    let mut in_tests = false;
    let mut printed_header = false;
    for (file, excerpts, omitted, is_test) in &files {
        if !printed_header {
            if !is_test {
                lines.push("Source files:".to_string());
            }
            printed_header = true;
        }
        if *is_test && !in_tests {
            lines.push("Test files (locations only; source not excerpted):".to_string());
            in_tests = true;
        }
        let roles = if file.roles.is_empty() {
            "relevant; role uncertain".to_string()
        } else {
            file.roles.join(", ")
        };
        let evidence = if *is_test {
            "locations only"
        } else if !excerpts.is_empty() {
            "source below"
        } else if *omitted {
            "source omitted"
        } else {
            "locations only"
        };
        lines.push(format!("- {} — {}; {}", quote(&file.path), roles, evidence));
    }
    lines.push("End file list. Declaration locations follow source.".to_string());

    for (file, excerpts, _, is_test) in &files {
        if *is_test {
            continue;
        }
        for excerpt in excerpts {
            push_excerpt(&mut lines, &file.path, excerpt);
        }
    }

    lines.push(String::new());
    lines.push("Declaration locations:".to_string());
    for (file, _, omitted, _) in &files {
        lines.push(format!("- {}", quote(&file.path)));
        let mut leads: Vec<&crate::types::ReadingLead> = file.leads.iter().collect();
        leads.sort_by_key(|lead| lead.range.start_line);
        for lead in leads {
            lines.push(format!(
                "  {}@{}-{}",
                lead.name, lead.range.start_line, lead.range.end_line
            ));
        }
        for call in &file.call_leads {
            lines.push(format!(
                "  Possible local call {} -> {}: lines {}-{}; runtime dispatch not verified.",
                call.caller, call.name, call.range.start_line, call.range.end_line
            ));
        }
        if *omitted {
            lines.push("  Some source omitted; locations remain available.".to_string());
        }
    }
    lines.push(String::new());
    lines.push("End context.".to_string());
    lines.join("\n") + "\n"
}


/// Structured output for programmatic consumers. Schema version 1, camelCase.
pub fn render_json(result: &RetrievalResult) -> String {
    let files: Vec<serde_json::Value> = result
        .files
        .iter()
        .map(|file| {
            serde_json::json!({
                "path": file.path,
                "score": (file.score * 10000.0).round() / 10000.0,
                "roles": file.roles,
                "priority": file.priority,
                "testFile": is_test_path(&file.path),
                "leads": file.leads.iter().map(|lead| serde_json::json!({
                    "name": lead.name,
                    "startLine": lead.range.start_line,
                    "endLine": lead.range.end_line,
                    "score": (lead.score * 10000.0).round() / 10000.0,
                })).collect::<Vec<_>>(),
                "callLeads": file.call_leads.iter().map(|call| serde_json::json!({
                    "caller": call.caller,
                    "name": call.name,
                    "startLine": call.range.start_line,
                    "endLine": call.range.end_line,
                })).collect::<Vec<_>>(),
                "excerpts": file.excerpts.iter().map(|excerpt| serde_json::json!({
                    "startLine": excerpt.range.start_line,
                    "endLine": excerpt.range.end_line,
                    "source": excerpt.source,
                })).collect::<Vec<_>>(),
                "sourceOmitted": file.source_omitted,
            })
        })
        .collect();
    let status = match result.status {
        Status::Complete => "complete",
        Status::Incomplete => "incomplete",
        Status::Interrupted => "interrupted",
    };
    let payload = serde_json::json!({
        "version": 1,
        "root": result.root,
        "query": result.query,
        "status": status,
        "files": files,
        "repositoryContext": {
            "instructionFiles": result.repository_context.instruction_files,
            "instructionLookupIncomplete": result.repository_context.instruction_lookup_incomplete,
            "pytestFiles": result.repository_context.pytest_files,
        },
        "issues": result.issues.iter().map(|(kind, count)| serde_json::json!({
            "kind": kind, "count": count,
        })).collect::<Vec<_>>(),
        "warnings": result.warnings.iter().map(|(kind, count)| serde_json::json!({
            "kind": kind, "count": count,
        })).collect::<Vec<_>>(),
        "counts": {
            "requests": result.counts.requests,
            "cacheHits": result.counts.cache_hits,
            "inspectedFiles": result.counts.inspected_files,
        },
        "providerFailure": result.provider_failure,
    });
    serde_json::to_string_pretty(&payload).unwrap_or_default() + "\n"
}

fn push_excerpt(lines: &mut Vec<String>, path: &str, excerpt: &crate::types::Excerpt) {
    lines.push(String::new());
    lines.push(format!(
        "Source block {} lines {}-{}:",
        quote(path),
        excerpt.range.start_line,
        excerpt.range.end_line
    ));
    let mut fence_length = 3;
    for match_ in excerpt.source.match_indices('`') {
        let backticks = excerpt.source[match_.0..]
            .chars()
            .take_while(|c| *c == '`')
            .count();
        fence_length = fence_length.max(backticks + 1);
    }
    let fence = "`".repeat(fence_length);
    lines.push(fence.clone());
    for line in excerpt.source.trim_end_matches('\n').split('\n') {
        lines.push(line.to_string());
    }
    lines.push(fence);
}

/// Evidence block for one file — excerpted source plus declaration and call
/// locations, in report vocabulary. Used by the TUI's print-on-quit so piped
/// output matches the text report's shapes.
pub fn file_evidence_text(file: &FileEvidence) -> String {
    let mut lines = Vec::new();
    let roles = if file.roles.is_empty() {
        "relevant; role uncertain".to_string()
    } else {
        file.roles.join(", ")
    };
    lines.push(format!("- {} — {}", quote(&file.path), roles));
    for excerpt in &file.excerpts {
        push_excerpt(&mut lines, &file.path, excerpt);
    }
    if !file.leads.is_empty() {
        lines.push(String::new());
        lines.push("  Declaration locations:".to_string());
        let mut leads: Vec<&crate::types::ReadingLead> = file.leads.iter().collect();
        leads.sort_by_key(|lead| lead.range.start_line);
        for lead in leads {
            lines.push(format!("  {}@{}-{}", lead.name, lead.range.start_line, lead.range.end_line));
        }
    }
    for call in &file.call_leads {
        lines.push(format!(
            "  Possible local call {} -> {}: lines {}-{}",
            call.caller, call.name, call.range.start_line, call.range.end_line
        ));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Excerpt, FileEvidence, ReadingLead};

    fn file(path: &str, score: f64) -> FileEvidence {
        FileEvidence {
            path: path.to_string(),
            score,
            roles: vec![],
            priority: None,
            leads: vec![ReadingLead {
                name: "unit".to_string(),
                range: crate::source::Range { start_line: 1, end_line: 2 },
                score: 0.5,
            }],
            call_leads: vec![],
            excerpts: vec![Excerpt {
                range: crate::source::Range { start_line: 1, end_line: 2 },
                source: "fn unit() {}".to_string(),
            }],
            source_omitted: false,
        }
    }

    fn result(files: Vec<FileEvidence>) -> RetrievalResult {
        RetrievalResult {
            root: ".".to_string(),
            query: "q".to_string(),
            status: Status::Complete,
            files,
            repository_context: Default::default(),
            issues: vec![],
            warnings: vec![],
            counts: Default::default(),
            provider_failure: None,
        }
    }

    #[test]
    fn test_paths_detected_conservatively() {
        for path in [
            "app/src/test/java/Foo.kt",
            "analytics/src/testFixtures/java/T.kt",
            "src/FooTest.kt",
            "src/AnalyticsTests.java",
            "src/FooIT.java",
            "src/test_foo.py",
            "src/foo_test.go",
            "src/foo.test.ts",
            "src/foo.spec.tsx",
            "src/__tests__/foo.js",
        ] {
            assert!(is_test_path(path), "should be test: {}", path);
        }
        for path in [
            "src/test_utils.rs",        // stem is test_utils, not *_test
            "src/latest.rs",            // contains 'test' but not a Test suffix
            "src/protractor.ts",       // ends 'tor', not '.test'
            "src/contest.py",
            "src/TestUtils.kt",        // Test prefix, not suffix — often main source
        ] {
            assert!(!is_test_path(path), "should NOT be test: {}", path);
        }
    }

    #[test]
    fn tests_group_after_source_and_never_dump_excerpts() {
        // Higher-scoring test must still render after lower-scoring source.
        let rendered = render_result(
            &result(vec![
                file("src/FooTest.kt", 0.9),
                file("src/prod.rs", 0.5),
            ]),
            0,
        );
        let source_pos = rendered.find("src/prod.rs").expect("source listed");
        let test_pos = rendered.find("src/FooTest.kt").expect("test listed");
        assert!(source_pos < test_pos, "source before test");
        assert!(rendered.contains("Source files:"));
        assert!(rendered.contains("Test files (locations only; source not excerpted):"));
        assert!(rendered.contains("Source block \"src/prod.rs\""));
        assert!(!rendered.contains("Source block \"src/FooTest.kt\""), "no test excerpts");
        // Test locations still present.
        let locations = rendered.find("Declaration locations:").expect("locations");
        assert!(rendered[locations..].contains("src/FooTest.kt"));
        assert!(rendered.trim_end().ends_with("End context."));
    }

    #[test]
    fn all_tests_result_has_no_source_header() {
        let rendered = render_result(&result(vec![file("src/a_test.go", 0.9)]), 0);
        assert!(!rendered.contains("Source files:"));
        assert!(rendered.contains("Test files (locations only; source not excerpted):"));
    }

    #[test]
    fn summary_counts_tests_and_source() {
        let rendered = render_result(
            &result(vec![file("src/prod.rs", 0.9), file("src/FooTest.kt", 0.8)]),
            0,
        );
        assert!(
            rendered.starts_with("Layagrep: 2 relevant files · 1 with source · 1 test files."),
            "{}",
            rendered.lines().next().unwrap()
        );
    }

    #[test]
    fn json_marks_test_files_and_priority() {
        let payload = render_json(&result(vec![
            file("src/prod.rs", 0.9),
            file("src/FooTest.kt", 0.8),
        ]));
        assert!(payload.contains("\"testFile\": true"));
        assert!(payload.contains("\"testFile\": false"));
        assert!(payload.contains("\"priority\""));
    }
}
