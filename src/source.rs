//! Source unit extraction: declarations with qualified names and owner class
//! headers, mirroring jevgrep's unit model. Python and TypeScript/JavaScript
//! parse through tree-sitter; everything else falls back to bounded text
//! chunks. Comment ranges feed excerpt window expansion.

use crate::walk::Snapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Range {
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone)]
pub struct SourceUnit {
    pub name: String,
    pub range: Range,
    pub byte_start: usize,
    pub byte_end: usize,
    pub owner_headers: Vec<Range>,
    pub partial: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Python,
    TypeScript,
    Text,
}

#[derive(Debug, Clone)]
pub struct Inspection {
    pub units: Vec<SourceUnit>,
    pub comments: Vec<Range>,
    pub mode: Mode,
}

pub const MAX_PARSE_BYTES: usize = 1_000_000;
const TEXT_CHUNK_BYTES: usize = 2_500;

pub struct LineIndex {
    /// Byte offset of the start of each line (line i+1 starts at offsets[i]).
    offsets: Vec<usize>,
    pub line_count: usize,
}

impl LineIndex {
    pub fn new(source: &str) -> Self {
        let mut offsets = vec![0usize];
        for (index, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                offsets.push(index + 1);
            }
        }
        let line_count = offsets.len();
        LineIndex { offsets, line_count }
    }

    /// 1-based line containing the given byte offset.
    pub fn line_of(&self, byte: usize) -> u32 {
        match self.offsets.binary_search(&byte) {
            Ok(position) => position as u32 + 1,
            Err(insertion) => insertion as u32,
        }
    }

    /// Byte offset of the start of a 1-based line, clamped to the source length.
    pub fn line_start(&self, line: u32) -> usize {
        if line == 0 {
            return 0;
        }
        self.offsets
            .get((line - 1) as usize)
            .copied()
            .unwrap_or_else(|| *self.offsets.last().unwrap_or(&0))
    }

    /// Byte offset of the end (exclusive, before the newline) of a 1-based line.
    pub fn line_end(&self, line: u32) -> usize {
        if (line as usize) < self.offsets.len() {
            self.offsets[line as usize] - 1
        } else {
            self.line_start(line).max(*self.offsets.last().unwrap_or(&1))
        }
    }
}

fn is_python(path: &str) -> bool {
    path.ends_with(".py") || path.ends_with(".pyi")
}

fn script_kind(path: &str) -> Option<&'static str> {
    if path.ends_with(".ts") || path.ends_with(".mts") || path.ends_with(".cts") {
        Some("typescript")
    } else if path.ends_with(".tsx") || path.ends_with(".jsx") {
        Some("tsx")
    } else if path.ends_with(".js") || path.ends_with(".mjs") || path.ends_with(".cjs") {
        Some("javascript")
    } else if path.ends_with(".rs") {
        Some("rust")
    } else if path.ends_with(".go") {
        Some("go")
    } else {
        None
    }
}

fn parse(language: tree_sitter::Language, source: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(source, None)?;
    if tree.root_node().has_error() {
        return None;
    }
    Some(tree)
}

pub fn inspect(snapshot: &Snapshot) -> Inspection {
    let index = LineIndex::new(&snapshot.source);
    if snapshot.bytes > MAX_PARSE_BYTES {
        return Inspection {
            units: text_units(&snapshot.source, &index),
            comments: Vec::new(),
            mode: Mode::Text,
        };
    }
    let path = &snapshot.path;
    if is_python(path) {
        let language: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
        if let Some(tree) = parse(language, &snapshot.source) {
            let mut units = Vec::new();
            let mut comments = Vec::new();
            python_visit(
                tree.root_node(),
                &snapshot.source,
                &index,
                "",
                &mut Vec::new(),
                &mut units,
                &mut comments,
            );
            comments.sort_by_key(|r| (r.start_line, r.end_line));
            comments.dedup_by_key(|r| *r);
            if !units.is_empty() || snapshot.source.trim().is_empty() {
                return Inspection { units, comments, mode: Mode::Python };
            }
        }
    } else if let Some(kind) = script_kind(path) {
        let language: tree_sitter::Language = match kind {
            "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
            "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "rust" => tree_sitter_rust::LANGUAGE.into(),
            "go" => tree_sitter_go::LANGUAGE.into(),
            _ => tree_sitter_javascript::LANGUAGE.into(),
        };
        if let Some(tree) = parse(language, &snapshot.source) {
            let mut units = Vec::new();
            let mut comments = Vec::new();
            match kind {
                "rust" => rust_visit(
                    tree.root_node(),
                    &snapshot.source,
                    &index,
                    "",
                    &mut Vec::new(),
                    &mut units,
                    &mut comments,
                ),
                "go" => go_visit(
                    tree.root_node(),
                    &snapshot.source,
                    &index,
                    &mut units,
                    &mut comments,
                ),
                _ => script_visit(
                    tree.root_node(),
                    &snapshot.source,
                    &index,
                    "",
                    &mut Vec::new(),
                    &mut units,
                    &mut comments,
                ),
            }
            comments.sort_by_key(|r| (r.start_line, r.end_line));
            comments.dedup_by_key(|r| *r);
            if !units.is_empty() || snapshot.source.trim().is_empty() {
                return Inspection { units, comments, mode: Mode::TypeScript };
            }
        }
    }
    Inspection {
        units: text_units(&snapshot.source, &index),
        comments: Vec::new(),
        mode: Mode::Text,
    }
}

// ---------------------------------------------------------------------------

fn node_range(node: tree_sitter::Node, source: &str, index: &LineIndex) -> Range {
    let start_line = index.line_of(node.start_byte());
    // The node's last occupied character, matching jevgrep's `end - 1` rule.
    let end_byte = node.end_byte().saturating_sub(1).min(source.len().saturating_sub(1));
    let mut end_line = index.line_of(end_byte);
    if end_line < start_line {
        end_line = start_line;
    }
    Range { start_line, end_line }
}

fn named_child_text<'a>(node: tree_sitter::Node<'a>, source: &'a str, field: &str) -> Option<&'a str> {
    node.child_by_field_name(field).map(|child| &source[child.byte_range()])
}

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------

fn python_visit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    prefix: &str,
    owner_headers: &mut Vec<Range>,
    units: &mut Vec<SourceUnit>,
    comments: &mut Vec<Range>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "comment" => comments.push(node_range(child, source, index)),
            "decorated_definition" => {
                if let Some(inner) = child.child_by_field_name("definition") {
                    let range = node_range(child, source, index);
                    let name = named_child_text(inner, source, "name")
                        .unwrap_or("anonymous")
                        .to_string();
                    units.push(SourceUnit {
                        name: format!("{}{}", prefix, name),
                        range,
                        byte_start: child.start_byte(),
                        byte_end: child.end_byte(),
                        owner_headers: owner_headers.clone(),
                        partial: false,
                    });
                }
            }
            "function_definition" => {
                let range = node_range(child, source, index);
                let name = named_child_text(child, source, "name")
                    .unwrap_or("anonymous")
                    .to_string();
                units.push(SourceUnit {
                    name: format!("{}{}", prefix, name),
                    range,
                    byte_start: child.start_byte(),
                    byte_end: child.end_byte(),
                    owner_headers: owner_headers.clone(),
                    partial: false,
                });
            }
            "class_definition" => {
                let name = named_child_text(child, source, "name")
                    .unwrap_or("Anonymous")
                    .to_string();
                let qualified = format!("{}{}", prefix, name);
                let class_range = node_range(child, source, index);
                let mut body_first_line = None;
                if let Some(body) = child.child_by_field_name("body") {
                    if let Some(first) = body.named_child(0) {
                        body_first_line = Some(node_range(first, source, index).start_line);
                    }
                }
                if let Some(first_body_line) = body_first_line {
                    if first_body_line > class_range.start_line {
                        let header = Range {
                            start_line: class_range.start_line,
                            end_line: first_body_line - 1,
                        };
                        units.push(SourceUnit {
                            name: format!("{}.context", qualified),
                            range: header,
                            byte_start: child.start_byte(),
                            byte_end: index.line_end(header.end_line) + 1,
                            owner_headers: owner_headers.clone(),
                            partial: false,
                        });
                        owner_headers.push(header);
                        python_visit(
                            child.child_by_field_name("body").unwrap(),
                            source,
                            index,
                            &format!("{}.", qualified),
                            owner_headers,
                            units,
                            comments,
                        );
                        owner_headers.pop();
                    }
                }
                // A class without a separable header keeps a whole-class unit.
                if body_first_line.is_none() {
                    units.push(SourceUnit {
                        name: qualified.clone(),
                        range: class_range,
                        byte_start: child.start_byte(),
                        byte_end: child.end_byte(),
                        owner_headers: owner_headers.clone(),
                        partial: false,
                    });
                }
            }
            "expression_statement" | "assignment" if prefix.is_empty() => {
                // Top-level simple assignments become named units (constants).
                if let Some(assignment) = child.named_child(0) {
                    if assignment.kind() == "assignment" {
                        if let Some(target) = assignment.child_by_field_name("left") {
                            if target.kind() == "identifier" {
                                let range = node_range(child, source, index);
                                units.push(SourceUnit {
                                    name: source[target.byte_range()].to_string(),
                                    range,
                                    byte_start: child.start_byte(),
                                    byte_end: child.end_byte(),
                                    owner_headers: owner_headers.clone(),
                                    partial: false,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// TypeScript / JavaScript
// ---------------------------------------------------------------------------

fn script_visit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    prefix: &str,
    owner_headers: &mut Vec<Range>,
    units: &mut Vec<SourceUnit>,
    comments: &mut Vec<Range>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_statement(child, source, index, prefix, owner_headers, units, comments);
    }
}

/// Match one statement-shaped node. `export` re-dispatches its inner
/// declaration here; container nodes are iterated by `script_visit`.
#[allow(clippy::too_many_arguments)]
fn visit_statement(
    child: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    prefix: &str,
    owner_headers: &mut Vec<Range>,
    units: &mut Vec<SourceUnit>,
    comments: &mut Vec<Range>,
) {
    {
        match child.kind() {
            "comment" => comments.push(node_range(child, source, index)),
            "class_declaration" => {
                let name = named_child_text(child, source, "name")
                    .unwrap_or("Anonymous")
                    .to_string();
                let qualified = format!("{}{}", prefix, name);
                let class_range = node_range(child, source, index);
                if let Some(body) = child.child_by_field_name("body") {
                    let mut header = None;
                    if let Some(first) = body.named_child(0) {
                        let first_line = node_range(first, source, index).start_line;
                        if first_line > class_range.start_line {
                            header = Some(Range {
                                start_line: class_range.start_line,
                                end_line: first_line - 1,
                            });
                        }
                    }
                    if let Some(header_range) = header {
                        units.push(SourceUnit {
                            name: format!("{}.context", qualified),
                            range: header_range,
                            byte_start: child.start_byte(),
                            byte_end: index.line_end(header_range.end_line) + 1,
                            owner_headers: owner_headers.clone(),
                            partial: false,
                        });
                        owner_headers.push(header_range);
                        script_class_members(body, source, index, &qualified, owner_headers, units);
                        owner_headers.pop();
                    } else {
                        script_class_members(body, source, index, &qualified, owner_headers, units);
                    }
                }
            }
            "export_statement" => {
                if let Some(declaration) = child.named_child(0) {
                    visit_statement(declaration, source, index, prefix, owner_headers, units, comments);
                    return;
                }
                units.push(statement_unit(child, source, index, prefix, "source", owner_headers));
            }
            "lexical_declaration" | "variable_declaration" => {
                let mut names = Vec::new();
                if let Some(list) = child.named_child(0) {
                    if list.kind() == "variable_declarator" {
                        if let Some(name_node) = list.child_by_field_name("name") {
                            collect_binding_names(name_node, source, &mut names);
                        }
                    } else {
                        let mut list_cursor = list.walk();
                        for declarator in list.children(&mut list_cursor) {
                            if declarator.kind() == "variable_declarator" {
                                if let Some(name_node) = declarator.child_by_field_name("name") {
                                    collect_binding_names(name_node, source, &mut names);
                                }
                            }
                        }
                    }
                }
                let name = if names.is_empty() {
                    "source".to_string()
                } else {
                    names.join(", ")
                };
                units.push(statement_unit(child, source, index, prefix, &name, owner_headers));
            }
            "function_declaration" | "generator_function_declaration" | "function_signature"
            | "abstract_class_declaration" | "enum_declaration" | "interface_declaration"
            | "type_alias_declaration" | "module_declaration" | "import_statement" => {
                let name = named_child_text(child, source, "name")
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "source".to_string());
                units.push(statement_unit(child, source, index, prefix, &name, owner_headers));
            }
            "expression_statement" | "if_statement" | "for_statement" | "for_in_statement"
            | "while_statement" | "switch_statement" | "try_statement" | "return_statement"
            | "throw_statement" | "block" if prefix.is_empty() => {
                units.push(statement_unit(child, source, index, prefix, "source", owner_headers));
            }
            _ => {
                if prefix.is_empty() && child.is_named() {
                    units.push(statement_unit(child, source, index, prefix, "source", owner_headers));
                }
            }
        }
    }
}

fn script_class_members(
    body: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    qualified: &str,
    owner_headers: &mut Vec<Range>,
    units: &mut Vec<SourceUnit>,
) {
    let mut cursor = body.walk();
    for member in body.children(&mut cursor) {
        let name = match member.kind() {
            "method_definition" | "public_field_definition" | "property_signature"
            | "abstract_method_signature" | "field_definition" => {
                named_child_text(member, source, "name").map(|n| n.to_string())
            }
            "function_declaration" => {
                named_child_text(member, source, "name").map(|n| n.to_string())
            }
            _ => None,
        };
        let name = match name {
            Some(name) => name,
            None => continue,
        };
        units.push(SourceUnit {
            name: format!("{}.{}", qualified, name),
            range: node_range(member, source, index),
            byte_start: member.start_byte(),
            byte_end: member.end_byte(),
            owner_headers: owner_headers.clone(),
            partial: false,
        });
    }
}

fn collect_binding_names(node: tree_sitter::Node, source: &str, names: &mut Vec<String>) {
    match node.kind() {
        "identifier" => names.push(source[node.byte_range()].to_string()),
        "object_pattern" | "array_pattern" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_binding_names(child, source, names);
            }
        }
        "shorthand_property_identifier_pattern" => {
            names.push(source[node.byte_range()].to_string());
        }
        _ => {}
    }
}

fn statement_unit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    _prefix: &str,
    name: &str,
    owner_headers: &[Range],
) -> SourceUnit {
    SourceUnit {
        name: name.to_string(),
        range: node_range(node, source, index),
        byte_start: node.start_byte(),
        byte_end: node.end_byte(),
        owner_headers: owner_headers.to_vec(),
        partial: false,
    }
}

// ---------------------------------------------------------------------------
// Text fallback
// ---------------------------------------------------------------------------

pub fn text_units(source: &str, index: &LineIndex) -> Vec<SourceUnit> {
    let mut units = Vec::new();
    if source.is_empty() {
        return units;
    }
    let total = source.len();
    let mut start = 0usize;
    let mut start_line = 1u32;
    while start < total {
        let mut end = (start + TEXT_CHUNK_BYTES).min(total);
        if end < total {
            // Split on a line boundary when one is reachable.
            if let Some(newline) = source[start..end].rfind('\n') {
                end = start + newline + 1;
            }
        }
        let end_line = index.line_of((end.saturating_sub(1)).min(total - 1));
        units.push(SourceUnit {
            name: "source".to_string(),
            range: Range { start_line, end_line: end_line.max(start_line) },
            byte_start: start,
            byte_end: end,
            owner_headers: Vec::new(),
            partial: true,
        });
        start = end;
        start_line = end_line + 1;
    }
    units
}

/// Source text of a line range, inclusive.
pub fn lines_text(source: &str, index: &LineIndex, range: Range) -> String {
    let start = index.line_start(range.start_line);
    let end = (index.line_end(range.end_line) + 1).min(source.len());
    source[start..end.max(start)].to_string()
}

// ---------------------------------------------------------------------------
// Rust
// ---------------------------------------------------------------------------

fn type_name_text<'a>(node: tree_sitter::Node<'a>, source: &'a str) -> Option<&'a str> {
    node.child_by_field_name("type").map(|t| &source[t.byte_range()])
}

fn rust_visit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    prefix: &str,
    owner_headers: &mut Vec<Range>,
    units: &mut Vec<SourceUnit>,
    comments: &mut Vec<Range>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "comment" => comments.push(node_range(child, source, index)),
            "function_item" | "function_signature_item" => {
                push_named_unit(child, source, index, prefix, "", owner_headers, units);
            }
            "struct_item" | "enum_item" | "union_item" | "trait_item" | "type_item" => {
                push_named_unit(child, source, index, prefix, "", owner_headers, units);
            }
            "const_item" | "static_item" if prefix.is_empty() => {
                push_named_unit(child, source, index, prefix, "", owner_headers, units);
            }
            "impl_item" => {
                let target = type_name_text(child, source)
                    .map(|t| t.trim().to_string())
                    .unwrap_or_else(|| "Impl".to_string());
                // `impl Trait for Foo` names the type; a bare type is itself.
                let qualified = format!("{}{}", prefix, target);
                let impl_range = node_range(child, source, index);
                if let Some(body) = child.child_by_field_name("body") {
                    let mut header = None;
                    if let Some(first) = body.named_child(0) {
                        let first_line = node_range(first, source, index).start_line;
                        if first_line > impl_range.start_line {
                            header = Some(Range {
                                start_line: impl_range.start_line,
                                end_line: first_line - 1,
                            });
                        }
                    }
                    if let Some(header_range) = header {
                        units.push(SourceUnit {
                            name: format!("{}.context", qualified),
                            range: header_range,
                            byte_start: child.start_byte(),
                            byte_end: index.line_end(header_range.end_line) + 1,
                            owner_headers: owner_headers.clone(),
                            partial: false,
                        });
                        owner_headers.push(header_range);
                        rust_visit(
                            body,
                            source,
                            index,
                            &format!("{}.", qualified),
                            owner_headers,
                            units,
                            comments,
                        );
                        owner_headers.pop();
                    } else {
                        rust_visit(
                            body,
                            source,
                            index,
                            &format!("{}.", qualified),
                            owner_headers,
                            units,
                            comments,
                        );
                    }
                }
            }
            "mod_item" => {
                let name = named_child_text(child, source, "name").unwrap_or("mod");
                rust_visit(
                    child,
                    source,
                    index,
                    &format!("{}{}::", prefix, name),
                    owner_headers,
                    units,
                    comments,
                );
            }
            _ => {}
        }
    }
}

fn push_named_unit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    prefix: &str,
    kind_label: &str,
    owner_headers: &[Range],
    units: &mut Vec<SourceUnit>,
) {
    if let Some(name) = named_child_text(node, source, "name") {
        units.push(SourceUnit {
            name: format!("{}{}{}", prefix, kind_label, name),
            range: node_range(node, source, index),
            byte_start: node.start_byte(),
            byte_end: node.end_byte(),
            owner_headers: owner_headers.to_vec(),
            partial: false,
        });
    }
}

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------

/// Depth-first search for the receiver's type identifier, through pointer
/// types: `(s *Store)`.
fn find_type_identifier<'a>(node: tree_sitter::Node<'a>, source: &'a str) -> Option<&'a str> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "type_identifier" {
            return Some(&source[child.byte_range()]);
        }
        if let Some(found) = find_type_identifier(child, source) {
            return Some(found);
        }
    }
    let _ = cursor;
    None
}

fn go_visit(
    node: tree_sitter::Node,
    source: &str,
    index: &LineIndex,
    units: &mut Vec<SourceUnit>,
    comments: &mut Vec<Range>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "comment" => comments.push(node_range(child, source, index)),
            "function_declaration" => {
                if let Some(name) = named_child_text(child, source, "name") {
                    units.push(SourceUnit {
                        name: name.to_string(),
                        range: node_range(child, source, index),
                        byte_start: child.start_byte(),
                        byte_end: child.end_byte(),
                        owner_headers: Vec::new(),
                        partial: false,
                    });
                }
            }
            "method_declaration" => {
                let name = named_child_text(child, source, "name").unwrap_or("Method");
                let receiver = child
                    .child_by_field_name("receiver")
                    .and_then(|receiver| find_type_identifier(receiver, source))
                    .unwrap_or("Recv");
                units.push(SourceUnit {
                    name: format!("{}.{}", receiver.trim_start_matches('*'), name),
                    range: node_range(child, source, index),
                    byte_start: child.start_byte(),
                    byte_end: child.end_byte(),
                    owner_headers: Vec::new(),
                    partial: false,
                });
            }
            "type_declaration" => {
                let mut cursor = child.walk();
                for spec in child.children(&mut cursor) {
                    if spec.kind() == "type_spec" {
                        if let Some(name) = named_child_text(spec, source, "name") {
                            units.push(SourceUnit {
                                name: name.to_string(),
                                range: node_range(spec, source, index),
                                byte_start: spec.start_byte(),
                                byte_end: spec.end_byte(),
                                owner_headers: Vec::new(),
                                partial: false,
                            });
                        }
                    }
                }
            }
            "const_declaration" | "var_declaration" => {
                let mut cursor = child.walk();
                for spec in child.children(&mut cursor) {
                    if spec.kind() == "const_spec" || spec.kind() == "var_spec" {
                        let mut inner = spec.walk();
                        let mut names: Vec<&str> = Vec::new();
                        for identifier in spec.children(&mut inner) {
                            if identifier.kind() == "identifier" {
                                names.push(&source[identifier.byte_range()]);
                            }
                        }
                        if !names.is_empty() {
                            units.push(SourceUnit {
                                name: names.join(", "),
                                range: node_range(spec, source, index),
                                byte_start: spec.start_byte(),
                                byte_end: spec.end_byte(),
                                owner_headers: Vec::new(),
                                partial: false,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
