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

/// Present source before detailed reading leads so truncated output remains useful.
pub fn render_result(result: &RetrievalResult, max_source_bytes: usize) -> String {
    let mut remaining = if max_source_bytes > 0 {
        max_source_bytes
    } else {
        usize::MAX
    };
    let mut files: Vec<(&FileEvidence, Vec<&crate::types::Excerpt>, bool)> = Vec::new();
    let mut ordered: Vec<&FileEvidence> = result.files.iter().collect();
    ordered.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    for file in ordered {
        let mut omitted = file.source_omitted;
        let excerpts: Vec<&crate::types::Excerpt> = file
            .excerpts
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
            .collect();
        files.push((file, excerpts, omitted));
    }

    let context = &result.repository_context;
    let omitted_count = files.iter().filter(|(_, _, omitted)| *omitted).count();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "Layagrep: {} relevant files{}.",
        files.len(),
        if result.status != Status::Complete { "; discovery incomplete" } else { "" }
    ));
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
    for (file, excerpts, omitted) in &files {
        let roles = if file.roles.is_empty() {
            "relevant; role uncertain".to_string()
        } else {
            file.roles.join(", ")
        };
        let evidence = if !excerpts.is_empty() {
            "source below"
        } else if *omitted {
            "source omitted"
        } else {
            "locations only"
        };
        lines.push(format!("- {} — {}; {}", quote(&file.path), roles, evidence));
    }
    lines.push("End file list. Declaration locations follow source.".to_string());

    for (file, excerpts, _) in &files {
        for excerpt in excerpts {
            lines.push(String::new());
            lines.push(format!(
                "Source block {} lines {}-{}:",
                quote(&file.path),
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
    }

    lines.push(String::new());
    lines.push("Declaration locations:".to_string());
    for (file, _, omitted) in &files {
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
