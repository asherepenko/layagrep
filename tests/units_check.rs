#[path = "../src/walk.rs"]
mod walk;
#[path = "../src/source.rs"]
mod source;

#[test]
fn export_naming() {
    let text = "import { x } from \"y\";\n\nexport type Thing = { a: string };\n\nexport function makeThing(): Thing {\n  return { a: \"b\" };\n}\n";
    let snapshot = walk::Snapshot {
        path: "mod.ts".into(),
        source: text.to_string(),
        content_hash: String::new(),
        bytes: text.len(),
    };
    let inspection = source::inspect(&snapshot);
    let names: Vec<String> = inspection
        .units
        .iter()
        .map(|u| format!("{}@{}-{}", u.name, u.range.start_line, u.range.end_line))
        .collect();
    eprintln!("UNITS: {:?}", names);
    assert!(names.iter().any(|n| n.starts_with("makeThing")), "units: {:?}", names);
    assert!(names.iter().any(|n| n.starts_with("Thing")), "units: {:?}", names);
}
