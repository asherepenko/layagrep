//! Interactive results browser behind `layagrep tui`.
//!
//! Search runs on a worker thread; progress streams into the status line.
//! Layout: header (query + summary), file list grouped source-then-tests
//! (same structure as the text report), detail pane with excerpts and
//! declaration locations for the selected file, footer with keys.
//! `Space` marks files; `q` quits and prints marked evidence to stdout
//! (piped consumers get report-vocabulary text); `Esc` quits silently.

use std::collections::HashSet;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::DefaultTerminal;

use crate::render::{file_evidence_text, grouped_files, is_test_path};
use crate::retrieve::{self, Progress, ProgressEvent, SPINNER_FRAMES};
use crate::types::{FileEvidence, RetrievalResult, Status};
use crate::walk::Policy;

/// Message from the search thread to the UI.
pub enum SearchMessage {
    Done(Box<RetrievalResult>),
    Crashed(String),
}

enum Row {
    Header(&'static str),
    File(usize),
}

pub struct TuiApp {
    query: String,
    rows: Vec<Row>,
    files: Vec<FileEvidence>,
    list_state: ListState,
    marked: HashSet<String>,
    detail_scroll: u16,
    searching: Option<String>,
    error: Option<String>,
    ticks: usize,
    status: Status,
    no_color: bool,
    pub quit: bool,
    pub print_marked: bool,
}

impl TuiApp {
    pub fn new(query: &str, no_color: bool) -> Self {
        TuiApp {
            query: query.to_string(),
            rows: Vec::new(),
            files: Vec::new(),
            list_state: ListState::default(),
            marked: HashSet::new(),
            detail_scroll: 0,
            searching: Some("starting".to_string()),
            error: None,
            ticks: 0,
            status: Status::Complete,
            no_color,
            quit: false,
            print_marked: false,
        }
    }

    pub fn update_progress(&mut self, event: &ProgressEvent) {
        self.searching = Some(if event.files_total > 0 {
            format!("{} {}/{} files", event.stage, event.files_done, event.files_total)
        } else {
            event.stage.to_string()
        });
    }

    pub fn set_result(&mut self, result: RetrievalResult) {
        self.searching = None;
        self.status = result.status;
        let (main_files, test_files) = grouped_files(&result);
        let mut rows = Vec::new();
        let mut files = Vec::new();
        if !main_files.is_empty() {
            rows.push(Row::Header("Source files"));
        }
        for file in main_files {
            rows.push(Row::File(files.len()));
            files.push(file.clone());
        }
        if !test_files.is_empty() {
            rows.push(Row::Header("Test files (locations only)"));
        }
        for file in test_files {
            rows.push(Row::File(files.len()));
            files.push(file.clone());
        }
        self.files = files;
        self.rows = rows;
        if self.file_rows() > 0 {
            self.list_state.select(Some(self.first_file_row()));
        }
    }

    pub fn set_error(&mut self, message: String) {
        self.searching = None;
        self.error = Some(message);
    }

    fn file_rows(&self) -> usize {
        self.files.len()
    }

    fn first_file_row(&self) -> usize {
        self.rows
            .iter()
            .position(|row| matches!(row, Row::File(_)))
            .unwrap_or(0)
    }

    fn last_file_row(&self) -> usize {
        self.rows
            .iter()
            .rposition(|row| matches!(row, Row::File(_)))
            .unwrap_or(0)
    }

    fn selected_row(&self) -> Option<usize> {
        self.list_state.selected()
    }

    fn selected_file(&self) -> Option<&FileEvidence> {
        let row_index = self.selected_row()?;
        match self.rows.get(row_index)? {
            Row::File(index) => self.files.get(*index),
            Row::Header(_) => None,
        }
    }

    fn move_selection(&mut self, forward: bool) {
        if self.rows.is_empty() {
            return;
        }
        let current = self.selected_row().unwrap_or(0);
        let next = if forward {
            (current + 1..self.rows.len())
                .find(|index| matches!(self.rows[*index], Row::File(_)))
                .unwrap_or(current)
        } else {
            (0..current)
                .rev()
                .find(|index| matches!(self.rows[*index], Row::File(_)))
                .unwrap_or(current)
        };
        self.list_state.select(Some(next));
        self.detail_scroll = 0;
    }

    fn jump_to_edge(&mut self, last: bool) {
        if self.rows.is_empty() {
            return;
        }
        let target = if last { self.last_file_row() } else { self.first_file_row() };
        self.list_state.select(Some(target));
        self.detail_scroll = 0;
    }

    fn scroll_detail(&mut self, down: bool) {
        self.detail_scroll = if down {
            self.detail_scroll.saturating_add(10)
        } else {
            self.detail_scroll.saturating_sub(10)
        };
    }

    fn toggle_mark(&mut self) {
        let path = self.selected_file().map(|file| file.path.clone());
        if let Some(path) = path {
            if !self.marked.insert(path.clone()) {
                self.marked.remove(&path);
            }
        }
    }

    pub fn on_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        match code {
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(true),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(false),
            KeyCode::Char('g') => self.jump_to_edge(false),
            KeyCode::Char('G') => self.jump_to_edge(true),
            KeyCode::PageDown | KeyCode::Char('J') => self.scroll_detail(true),
            KeyCode::PageUp | KeyCode::Char('K') => self.scroll_detail(false),
            KeyCode::Char(' ') => self.toggle_mark(),
            KeyCode::Enter => self.detail_scroll = 0,
            KeyCode::Esc => {
                self.quit = true;
            }
            KeyCode::Char('q') => {
                self.quit = true;
                self.print_marked = !self.marked.is_empty();
            }
            _ => {}
        }
    }

    fn style(&self, base: Style) -> Style {
        if self.no_color {
            Style::new()
        } else {
            base
        }
    }

    fn list_items(&self) -> Vec<ListItem<'static>> {
        let mut items = Vec::new();
        for row in &self.rows {
            match row {
                Row::Header(title) => {
                    let style = self.style(Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD));
                    items.push(ListItem::new(Line::styled(format!(" {} ", title), style)));
                }
                Row::File(index) => {
                    let file = &self.files[*index];
                    let is_test = is_test_path(&file.path);
                    let marked = self.marked.contains(&file.path);
                    let mut spans = Vec::new();
                    spans.push(Span::styled(
                        if marked { "◆ " } else { "  " },
                        self.style(Style::new().fg(Color::Green).bold()),
                    ));
                    spans.push(Span::styled(
                        format!("{:.2} ", file.score),
                        self.style(Style::new().fg(Color::Cyan).dim()),
                    ));
                    let (dir, name) = match file.path.rsplit_once('/') {
                        Some((dir, name)) => (format!("{}/", dir), name.to_string()),
                        None => (String::new(), file.path.clone()),
                    };
                    let file_style = if is_test {
                        self.style(Style::new().dim())
                    } else {
                        self.style(Style::new().bold())
                    };
                    spans.push(Span::styled(dir, self.style(Style::new().dim())));
                    spans.push(Span::styled(name, file_style));
                    let flags = [
                        (!file.excerpts.is_empty()).then_some("src"),
                        (is_test).then_some("test"),
                        (file.source_omitted).then_some("omitted"),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(",");
                    if !flags.is_empty() {
                        spans.push(Span::styled(
                            format!("  {}", flags),
                            self.style(Style::new().fg(Color::DarkGray)),
                        ));
                    }
                    items.push(ListItem::new(Line::from(spans)));
                }
            }
        }
        items
    }

    fn detail_lines(&self) -> Vec<Line<'static>> {
        let Some(file) = self.selected_file() else {
            return vec![Line::styled(
                self.error
                    .clone()
                    .unwrap_or_else(|| "No result yet.".to_string()),
                self.style(Style::new().fg(Color::Red)),
            )];
        };
        let mut lines = Vec::new();
        let is_test = is_test_path(&file.path);
        let roles = if file.roles.is_empty() {
            "relevant; role uncertain".to_string()
        } else {
            file.roles.join(", ")
        };
        lines.push(Line::from(vec![
            Span::styled(
                file.path.clone(),
                self.style(Style::new().fg(Color::White).bold()),
            ),
            Span::raw("  "),
            Span::styled(format!("{:.3}", file.score), self.style(Style::new().fg(Color::Cyan))),
            Span::raw("  "),
            Span::styled(roles, self.style(Style::new().fg(Color::Yellow).dim())),
        ]));
        let summary = if is_test {
            format!(
                "test file · {} declaration location(s) · source not excerpted",
                file.leads.len()
            )
        } else if file.excerpts.is_empty() {
            format!(
                "locations only · {} declaration location(s)",
                file.leads.len()
            )
        } else {
            format!(
                "{} excerpt(s) · {} declaration location(s)",
                file.excerpts.len(),
                file.leads.len()
            )
        };
        lines.push(Line::styled(summary, self.style(Style::new().dim())));
        lines.push(Line::raw(""));
        for excerpt in &file.excerpts {
            lines.push(Line::styled(
                format!("lines {}-{}", excerpt.range.start_line, excerpt.range.end_line),
                self.style(Style::new().fg(Color::Cyan).dim()),
            ));
            let mut number = excerpt.range.start_line;
            for source_line in excerpt.source.trim_end_matches('\n').split('\n') {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:>5} ", number),
                        self.style(Style::new().fg(Color::DarkGray)),
                    ),
                    Span::raw(source_line.to_string()),
                ]));
                number += 1;
            }
            lines.push(Line::raw(""));
        }
        if !file.excerpts.is_empty() {
            lines.push(Line::styled(
                "Declaration locations",
                self.style(Style::new().fg(Color::Cyan).dim()),
            ));
        }
        let mut leads: Vec<&crate::types::ReadingLead> = file.leads.iter().collect();
        leads.sort_by_key(|lead| lead.range.start_line);
        for lead in leads {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {}@", lead.name),
                    self.style(Style::new().fg(Color::White)),
                ),
                Span::raw(format!("{}-{}", lead.range.start_line, lead.range.end_line)),
            ]));
        }
        for call in &file.call_leads {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{} -> {}", call.caller, call.name), self.style(Style::new().dim())),
                Span::raw(format!(" (lines {}-{})", call.range.start_line, call.range.end_line)),
            ]));
        }
        lines
    }

    fn header_line(&self) -> Line<'static> {
        let mut spans = Vec::new();
        if let Some(stage) = &self.searching {
            let frame = SPINNER_FRAMES[self.ticks % SPINNER_FRAMES.len()];
            spans.push(Span::styled(
                format!("{} {}  ", frame, stage),
                self.style(Style::new().fg(Color::Magenta).bold()),
            ));
        } else if self.error.is_some() {
            spans.push(Span::styled(
                "search failed  ",
                self.style(Style::new().fg(Color::Red).bold()),
            ));
        } else {
            let incomplete = match self.status {
                Status::Complete => "",
                _ => "  ·  discovery incomplete",
            };
            spans.push(Span::styled(
                format!(
                    "{} file(s) · {} marked{}  ",
                    self.files.len(),
                    self.marked.len(),
                    incomplete
                ),
                self.style(Style::new().dim()),
            ));
        }
        spans.push(Span::styled(
            self.query.clone(),
            self.style(Style::new().fg(Color::Green).bold()),
        ));
        Line::from(spans)
    }

    pub fn draw<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut ratatui::Terminal<B>,
    ) -> std::io::Result<()> {
        self.ticks = self.ticks.wrapping_add(1);
        terminal
            .draw(|frame| {
            let area = frame.area();
            if area.width < 40 || area.height < 10 {
                let notice = Paragraph::new("terminal too small (need >= 40x10)")
                    .style(self.style(Style::new().fg(Color::Red)));
                frame.render_widget(notice, area);
                return;
            }
            let chunks = Layout::vertical([
                Constraint::Length(1),
                Constraint::Percentage(45),
                Constraint::Min(5),
                Constraint::Length(1),
            ])
            .split(area);

            frame.render_widget(Paragraph::new(self.header_line()), chunks[0]);

            let items = self.list_items();
            let list = List::new(items)
                .highlight_symbol("› ")
                .highlight_style(self.style(Style::new().bg(Color::DarkGray).add_modifier(Modifier::BOLD)));
            frame.render_stateful_widget(list, chunks[1], &mut self.list_state);

            let detail_lines = self.detail_lines();
            let total = detail_lines.len() as u16;
            let detail_text = Text::from(detail_lines);
            let visible_height = chunks[2].height.saturating_sub(2) as u16;
            let max_scroll = total.saturating_sub(visible_height);
            if self.detail_scroll > max_scroll {
                self.detail_scroll = max_scroll;
            }
            let detail = Paragraph::new(detail_text)
                .block(Block::bordered())
                .wrap(Wrap { trim: false })
                .scroll((self.detail_scroll, 0));
            frame.render_widget(detail, chunks[2]);

            let footer = Paragraph::new(Line::styled(
                "j/k select · Space mark · PgUp/PgDn scroll detail · g/G edges · q quit+print marked · Esc quit",
                self.style(Style::new().fg(Color::DarkGray)),
            ));
            frame.render_widget(footer, chunks[3]);
        })
        .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(())
    }

    /// Evidence text for marked files, in report vocabulary. Empty when
    /// nothing is marked.
    pub fn marked_evidence(&self) -> String {
        let mut parts = Vec::new();
        for file in &self.files {
            if self.marked.contains(&file.path) {
                parts.push(file_evidence_text(file));
            }
        }
        parts.join("")
    }
}

/// Public entry: run the search on a thread, drive the TUI, return the exit
/// code. Prints marked evidence to stdout after the screen is restored.
#[allow(clippy::too_many_arguments)]
pub fn run(
    query: &str,
    root: &str,
    policy: Policy,
    model: &str,
    dtype: &str,
    engine_choice: crate::judge::EngineKind,
    python: Option<&str>,
    workers: usize,
    no_cache: bool,
) -> Result<i32, String> {
    if !std::io::stdout().is_terminal() {
        return Err("tui requires an interactive terminal (stdout is not a tty)".to_string());
    }

    let (message_tx, message_rx) = mpsc::channel::<SearchMessage>();
    let (progress_tx, progress_rx) = mpsc::channel::<ProgressEvent>();
    let interrupt = Arc::new(AtomicBool::new(false));
    let interrupt_for_thread = Arc::clone(&interrupt);

    let query_owned = query.to_string();
    let root_owned = root.to_string();
    let model_for_judges = model.to_string();
    let model_for_engine = model.to_string();
    let dtype_owned = dtype.to_string();
    let python_owned = python.map(str::to_string);
    let handle = std::thread::spawn(move || {
        // Worker logs would corrupt the TUI: always quiet in tui mode.
        let workers = if workers > 0 {
            workers
        } else {
            match engine_choice {
                crate::judge::EngineKind::Native => 4,
                _ => 1,
            }
        };
        let factory: crate::engine::JudgeFactory = Box::new(move || {
            // JudgeFactory is FnMut (the engine may re-spawn): clone captures
            // here, then move the clones into the per-worker closure.
            let model = model_for_judges.clone();
            let dtype = dtype_owned.clone();
            let python = python_owned.clone();
            let pool = crate::judge::PoolJudge::spawn(workers, move |_| {
                crate::judge::create_judge(
                    engine_choice,
                    &model,
                    &dtype,
                    python.as_deref(),
                    true,
                )
            })?;
            Ok(Box::new(pool) as Box<dyn crate::judge::Judge>)
        });
        let cache = crate::cache::Cache::new(crate::worker::cache_dir(), !no_cache);
        let mut engine = crate::engine::Engine::new(&model_for_engine, factory, cache);
        let mut progress = Progress::reporting(progress_tx);
        let interrupted = move || interrupt_for_thread.load(Ordering::Relaxed);
        let options = retrieve::SearchOptions {
            policy,
            query: query_owned,
            root: root_owned,
            debug_scores: false,
        };
        let result = retrieve::retrieve(&options, &mut engine, &mut progress, &interrupted);
        let _ = message_tx.send(SearchMessage::Done(Box::new(result)));
    });

    let no_color = std::env::var_os("NO_COLOR").is_some();
    let mut app = TuiApp::new(query, no_color);
    let mut terminal = ratatui::init();
    let exit = event_loop(&mut terminal, &mut app, &message_rx, &progress_rx, &handle, &interrupt);
    ratatui::restore();

    if app.print_marked {
        print!("{}", app.marked_evidence());
    }
    Ok(exit)
}

fn event_loop(
    terminal: &mut DefaultTerminal,
    app: &mut TuiApp,
    messages: &Receiver<SearchMessage>,
    progress: &Receiver<ProgressEvent>,
    search: &std::thread::JoinHandle<()>,
    interrupt: &AtomicBool,
) -> i32 {
    loop {
        if app.draw(terminal).is_err() {
            return 1;
        }
        while let Ok(event) = progress.try_recv() {
            app.update_progress(&event);
        }
        match messages.try_recv() {
            Ok(SearchMessage::Done(result)) => app.set_result(*result),
            Ok(SearchMessage::Crashed(message)) => app.set_error(message),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {}
        }
        if app.error.is_none() && app.searching.is_some() && search.is_finished() {
            // Thread died without sending a result.
            app.set_error("search thread exited unexpectedly".to_string());
        }
        if event::poll(Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind == KeyEventKind::Press {
                    app.on_key(key.code, key.modifiers);
                }
            }
        }
        if app.quit {
            interrupt.store(true, Ordering::Relaxed);
            let interrupted_keys = app.error.is_some();
            if interrupted_keys {
                return 1;
            }
            return 0;
        }
    }
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

    fn populated_app() -> TuiApp {
        let mut app = TuiApp::new("password login?", true);
        app.set_result(result(vec![
            file("src/FooTest.kt", 0.9),
            file("src/prod.rs", 0.5),
            file("src/other.rs", 0.4),
        ]));
        app
    }

    #[test]
    fn result_groups_source_then_tests_with_headers() {
        let app = populated_app();
        let headers: Vec<&str> = app
            .rows
            .iter()
            .filter_map(|row| match row {
                Row::Header(title) => Some(*title),
                Row::File(_) => None,
            })
            .collect();
        assert_eq!(headers, vec!["Source files", "Test files (locations only)"]);
        // Selection starts on the first FILE row, not a header.
        let selected = app.selected_row().expect("selection");
        assert!(matches!(app.rows[selected], Row::File(_)));
        // First selected file is the top-scoring SOURCE file.
        assert_eq!(app.selected_file().unwrap().path, "src/prod.rs");
    }

    #[test]
    fn navigation_skips_headers_and_wraps_at_edges() {
        let mut app = populated_app();
        // prod.rs -> other.rs -> (no more source) -> FooTest.kt
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.selected_file().unwrap().path, "src/other.rs");
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.selected_file().unwrap().path, "src/FooTest.kt");
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.selected_file().unwrap().path, "src/FooTest.kt", "clamped at last file");
        app.on_key(KeyCode::Char('g'), KeyModifiers::NONE);
        assert_eq!(app.selected_file().unwrap().path, "src/prod.rs");
        app.on_key(KeyCode::Char('G'), KeyModifiers::NONE);
        assert_eq!(app.selected_file().unwrap().path, "src/FooTest.kt");
    }

    #[test]
    fn marking_and_printing_marked_evidence() {
        let mut app = populated_app();
        app.on_key(KeyCode::Char(' '), KeyModifiers::NONE); // mark prod.rs
        app.on_key(KeyCode::Down, KeyModifiers::NONE); // other.rs
        app.on_key(KeyCode::Char(' '), KeyModifiers::NONE); // mark other.rs
        assert_eq!(app.marked.len(), 2);
        let evidence = app.marked_evidence();
        assert!(evidence.contains("src/prod.rs"));
        assert!(evidence.contains("src/other.rs"));
        assert!(!evidence.contains("src/FooTest.kt"));
        assert!(evidence.contains("Source block \"src/prod.rs\" lines 1-2:"));
        // Unmark: Space again on other.rs
        app.on_key(KeyCode::Char(' '), KeyModifiers::NONE);
        assert_eq!(app.marked.len(), 1);
    }

    #[test]
    fn quit_semantics() {
        let mut app = populated_app();
        app.on_key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(app.quit);
        assert!(!app.print_marked, "nothing marked -> no print");

        let mut app = populated_app();
        app.on_key(KeyCode::Char(' '), KeyModifiers::NONE);
        app.on_key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(app.print_marked, "marked -> print on quit");

        let mut app = populated_app();
        app.on_key(KeyCode::Char(' '), KeyModifiers::NONE);
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.quit);
        assert!(!app.print_marked, "Esc never prints");
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = populated_app();
        app.on_key(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(app.quit);
        assert!(!app.print_marked);
    }

    #[test]
    fn detail_shows_excerpt_and_leads() {
        let app = populated_app();
        let text: Vec<String> = app
            .detail_lines()
            .iter()
            .map(|line| line.spans.iter().map(|span| span.content.to_string()).collect())
            .collect();
        let joined = text.join("\n");
        assert!(joined.contains("src/prod.rs"));
        assert!(joined.contains("lines 1-2"));
        assert!(joined.contains("fn unit() {}"));
        assert!(joined.contains("unit@1-2"));
    }

    #[test]
    fn renders_on_test_backend() {
        use ratatui::backend::TestBackend;
        let mut app = populated_app();
        let mut terminal = ratatui::Terminal::new(TestBackend::new(100, 30)).unwrap();
        app.draw(&mut terminal).unwrap();
        let buffer = terminal.backend().buffer();
        let content: String = buffer
            .content
            .iter()
            .map(|cell| cell.symbol().to_string())
            .collect();
        assert!(content.contains("Source files"));
        assert!(content.contains("prod.rs"));
        assert!(content.contains("FooTest.kt"));
        assert!(content.contains("password login?"));
        assert!(content.contains("q quit+print marked"));
    }

    #[test]
    fn searching_state_shows_stage_before_result() {
        let mut app = TuiApp::new("q", true);
        app.update_progress(&ProgressEvent {
            stage: "score",
            files_done: 5,
            files_total: 10,
        });
        assert_eq!(app.searching.as_deref(), Some("score 5/10 files"));
        let line = app.header_line();
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.contains("score 5/10 files"));
    }
}
