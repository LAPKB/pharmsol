use std::fmt;
use std::sync::Arc;

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn empty(offset: usize) -> Self {
        Self {
            start: offset,
            end: offset,
        }
    }

    pub fn join(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub const fn shifted(self, delta: usize) -> Self {
        Self {
            start: self.start + delta,
            end: self.end + delta,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosticCode {
    value: &'static str,
}

impl DiagnosticCode {
    pub const fn new(value: &'static str) -> Self {
        Self { value }
    }

    pub const fn as_str(self) -> &'static str {
        self.value
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.value)
    }
}

pub const DSL_PARSE_GENERIC: DiagnosticCode = DiagnosticCode::new("DSL1000");
pub const DSL_SEMANTIC_GENERIC: DiagnosticCode = DiagnosticCode::new("DSL2000");
pub const DSL_LOWERING_GENERIC: DiagnosticCode = DiagnosticCode::new("DSL3000");
pub const DSL_BACKEND_GENERIC: DiagnosticCode = DiagnosticCode::new("DSL4000");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticPhase {
    Parse,
    Semantic,
    Lowering,
    Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLabelKind {
    Primary,
    Secondary,
    Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLabel {
    pub kind: DiagnosticLabelKind,
    pub span: Span,
    pub message: Option<String>,
}

impl DiagnosticLabel {
    pub fn primary(span: Span) -> Self {
        Self {
            kind: DiagnosticLabelKind::Primary,
            span,
            message: None,
        }
    }

    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticLabelKind::Secondary,
            span,
            message: Some(message.into()),
        }
    }

    pub fn context(span: Span, message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticLabelKind::Context,
            span,
            message: Some(message.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    Always,
    MaybeIncorrect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    pub span: Span,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticSuggestion {
    pub message: String,
    pub edits: Vec<TextEdit>,
    pub applicability: Applicability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: DiagnosticSeverity,
    pub phase: DiagnosticPhase,
    pub message: String,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
    pub helps: Vec<String>,
    pub suggestions: Vec<DiagnosticSuggestion>,
}

impl Diagnostic {
    pub fn error(
        code: DiagnosticCode,
        phase: DiagnosticPhase,
        message: impl Into<String>,
        span: Span,
    ) -> Self {
        Self {
            code,
            severity: DiagnosticSeverity::Error,
            phase,
            message: message.into(),
            labels: vec![DiagnosticLabel::primary(span)],
            notes: Vec::new(),
            helps: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    pub fn primary_span(&self) -> Span {
        self.labels
            .iter()
            .find(|label| label.kind == DiagnosticLabelKind::Primary)
            .map(|label| label.span)
            .unwrap_or_default()
    }

    pub fn with_label(mut self, label: DiagnosticLabel) -> Self {
        self.labels.push(label);
        self
    }

    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel::secondary(span, message));
        self
    }

    pub fn with_context_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel::context(span, message));
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.helps.push(help.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: DiagnosticSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    pub fn shifted(mut self, delta: usize) -> Self {
        for label in &mut self.labels {
            label.span = label.span.shifted(delta);
        }
        for suggestion in &mut self.suggestions {
            for edit in &mut suggestion.edits {
                edit.span = edit.span.shifted(delta);
            }
        }
        self
    }

    pub fn render(&self, src: &str) -> String {
        let mut rendered = String::new();
        rendered.push_str(&format!(
            "{}[{}]: {}\n",
            self.severity.as_str(),
            self.code,
            self.message
        ));
        let primary = self.primary_span();
        let (line, column, _, _) = line_info(src, primary.start.min(src.len()));
        rendered.push_str(&format!("  --> line {}, column {}\n", line, column));

        if !self.labels.is_empty() {
            let gutter = self
                .labels
                .iter()
                .map(|label| line_info(src, label.span.start.min(src.len())).0)
                .max()
                .unwrap_or(line)
                .to_string()
                .len();
            rendered.push_str(&format!("{:>width$} |\n", "", width = gutter));
            for label in &self.labels {
                rendered.push_str(&render_label(src, label, &self.message, gutter));
            }
        }
        for note in &self.notes {
            rendered.push_str(&format!("  = note: {}\n", note,));
        }
        for help in &self.helps {
            rendered.push_str(&format!("  = help: {}\n", help,));
        }
        for suggestion in &self.suggestions {
            rendered.push_str(&format!("  = suggestion: {}\n", suggestion.message,));
        }
        rendered
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReport {
    pub source: DiagnosticReportSource,
    pub diagnostics: Vec<DiagnosticReportEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered: Option<String>,
}

impl DiagnosticReport {
    pub fn from_diagnostics(
        source_name: impl Into<String>,
        source_text: Option<&str>,
        diagnostics: &[Diagnostic],
    ) -> Self {
        Self {
            source: DiagnosticReportSource {
                name: source_name.into(),
                byte_len: source_text.map(str::len),
            },
            diagnostics: diagnostics
                .iter()
                .map(|diagnostic| DiagnosticReportEntry::from_diagnostic(diagnostic, source_text))
                .collect(),
            rendered: source_text.map(|src| render_diagnostics(diagnostics, src)),
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportSource {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportEntry {
    pub code: String,
    pub severity: String,
    pub phase: String,
    pub message: String,
    pub labels: Vec<DiagnosticReportLabel>,
    pub notes: Vec<String>,
    pub helps: Vec<String>,
    pub suggestions: Vec<DiagnosticReportSuggestion>,
}

impl DiagnosticReportEntry {
    fn from_diagnostic(diagnostic: &Diagnostic, source_text: Option<&str>) -> Self {
        Self {
            code: diagnostic.code.as_str().to_string(),
            severity: diagnostic.severity.as_str().to_string(),
            phase: diagnostic.phase.as_str().to_string(),
            message: diagnostic.message.clone(),
            labels: diagnostic
                .labels
                .iter()
                .map(|label| DiagnosticReportLabel::from_label(label, source_text))
                .collect(),
            notes: diagnostic.notes.clone(),
            helps: diagnostic.helps.clone(),
            suggestions: diagnostic
                .suggestions
                .iter()
                .map(|suggestion| {
                    DiagnosticReportSuggestion::from_suggestion(suggestion, source_text)
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportLabel {
    pub kind: String,
    pub span: DiagnosticReportSpan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl DiagnosticReportLabel {
    fn from_label(label: &DiagnosticLabel, source_text: Option<&str>) -> Self {
        Self {
            kind: label.kind.as_str().to_string(),
            span: DiagnosticReportSpan::from_span(label.span, source_text),
            message: label.message.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportSuggestion {
    pub message: String,
    pub edits: Vec<DiagnosticReportEdit>,
    pub applicability: String,
}

impl DiagnosticReportSuggestion {
    fn from_suggestion(suggestion: &DiagnosticSuggestion, source_text: Option<&str>) -> Self {
        Self {
            message: suggestion.message.clone(),
            edits: suggestion
                .edits
                .iter()
                .map(|edit| DiagnosticReportEdit::from_edit(edit, source_text))
                .collect(),
            applicability: suggestion.applicability.as_str().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportEdit {
    pub span: DiagnosticReportSpan,
    pub replacement: String,
}

impl DiagnosticReportEdit {
    fn from_edit(edit: &TextEdit, source_text: Option<&str>) -> Self {
        Self {
            span: DiagnosticReportSpan::from_span(edit.span, source_text),
            replacement: edit.replacement.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticReportSpan {
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

impl DiagnosticReportSpan {
    fn from_span(span: Span, source_text: Option<&str>) -> Self {
        let (start_line, start_column, end_line, end_column) = if let Some(src) = source_text {
            let (start_line, start_column, _, _) = line_info(src, span.start.min(src.len()));
            let (end_line, end_column, _, _) = line_info(src, span.end.min(src.len()));
            (
                Some(start_line),
                Some(start_column),
                Some(end_line),
                Some(end_column),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            start: span.start,
            end: span.end,
            start_line,
            start_column,
            end_line,
            end_column,
        }
    }
}

impl DiagnosticSeverity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
        }
    }
}

impl DiagnosticPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Semantic => "semantic",
            Self::Lowering => "lowering",
            Self::Backend => "backend",
        }
    }
}

impl DiagnosticLabelKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Secondary => "secondary",
            Self::Context => "context",
        }
    }
}

impl Applicability {
    fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::MaybeIncorrect => "maybe_incorrect",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ParseError {
    diagnostics: Vec<Diagnostic>,
    source: Option<Arc<str>>,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self::from_diagnostic(Diagnostic::error(
            DSL_PARSE_GENERIC,
            DiagnosticPhase::Parse,
            message,
            span,
        ))
    }

    pub fn from_diagnostic(diagnostic: Diagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
            source: None,
        }
    }

    pub fn from_diagnostics(diagnostics: Vec<Diagnostic>) -> Self {
        debug_assert!(
            !diagnostics.is_empty(),
            "parse errors must contain at least one diagnostic"
        );
        Self {
            diagnostics,
            source: None,
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.diagnostics[0].notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.diagnostics[0].helps.push(help.into());
        self
    }

    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostics[0]
            .labels
            .push(DiagnosticLabel::secondary(span, message));
        self
    }

    pub fn with_context_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostics[0]
            .labels
            .push(DiagnosticLabel::context(span, message));
        self
    }

    pub fn diagnostic(&self) -> &Diagnostic {
        &self.diagnostics[0]
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn into_diagnostic(self) -> Diagnostic {
        self.diagnostics
            .into_iter()
            .next()
            .expect("parse error diagnostic")
    }

    pub fn render(&self, src: &str) -> String {
        render_diagnostics(&self.diagnostics, src)
    }

    pub fn diagnostic_report(&self, source_name: impl Into<String>) -> DiagnosticReport {
        DiagnosticReport::from_diagnostics(source_name, self.source(), &self.diagnostics)
    }

    pub fn with_source(mut self, source: impl Into<Arc<str>>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    pub fn shifted(mut self, delta: usize) -> Self {
        for diagnostic in &mut self.diagnostics {
            *diagnostic = diagnostic.clone().shifted(delta);
        }
        self
    }
}

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = self.source() {
            return f.write_str(&self.render(source));
        }
        let span = self.diagnostic().primary_span();
        write!(
            f,
            "{} at bytes {}..{}",
            self.diagnostic().message,
            span.start,
            span.end
        )
    }
}

impl std::error::Error for ParseError {}

fn render_diagnostics(diagnostics: &[Diagnostic], src: &str) -> String {
    diagnostics
        .iter()
        .map(|diagnostic| diagnostic.render(src))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_label(
    src: &str,
    label: &DiagnosticLabel,
    fallback_message: &str,
    gutter: usize,
) -> String {
    let offset = label.span.start.min(src.len());
    let (line, _, line_start, line_end) = line_info(src, offset);
    let line_text = &src[line_start..line_end];
    let marker_start = src[line_start..offset].chars().count();
    let highlight_end = label.span.end.min(line_end).max(offset);
    let marker_len = src[offset..highlight_end].chars().count().max(1);
    let marker = match label.kind {
        DiagnosticLabelKind::Primary => '^',
        DiagnosticLabelKind::Secondary => '-',
        DiagnosticLabelKind::Context => '~',
    };
    let label_message = label.message.as_deref().unwrap_or(match label.kind {
        DiagnosticLabelKind::Primary => fallback_message,
        DiagnosticLabelKind::Secondary | DiagnosticLabelKind::Context => "",
    });

    let mut rendered = String::new();
    rendered.push_str(&format!(
        "{:>width$} | {}\n",
        line,
        line_text,
        width = gutter
    ));
    rendered.push_str(&format!(
        "{:>width$} | {}{}",
        "",
        " ".repeat(marker_start),
        marker.to_string().repeat(marker_len),
        width = gutter
    ));
    if !label_message.is_empty() {
        rendered.push_str(&format!(" {}", label_message));
    }
    rendered.push('\n');
    rendered
}

fn line_info(src: &str, offset: usize) -> (usize, usize, usize, usize) {
    let mut line = 1;
    let mut line_start = 0;
    for (idx, ch) in src.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + 1;
        }
    }

    let line_end = src[line_start..]
        .find('\n')
        .map(|idx| line_start + idx)
        .unwrap_or(src.len());
    let column = src[line_start..offset].chars().count() + 1;
    (line, column, line_start, line_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_includes_code_and_primary_label_message() {
        let src = "dx(gut) = 1 +\n";
        let diagnostic = Diagnostic::error(
            DSL_PARSE_GENERIC,
            DiagnosticPhase::Parse,
            "expected expression after `+`",
            Span::new(12, 13),
        );

        let rendered = diagnostic.render(src);
        assert!(
            rendered.contains("error[DSL1000]: expected expression after `+`"),
            "{}",
            rendered
        );
        assert!(rendered.contains("--> line 1, column 13"), "{}", rendered);
        assert!(
            rendered.contains("^ expected expression after `+`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn render_displays_secondary_and_context_labels() {
        let src = "model broken {\n  outputs {\n    cp = 1 +\n  }\n}\n";
        let diagnostic = Diagnostic::error(
            DSL_PARSE_GENERIC,
            DiagnosticPhase::Parse,
            "expected expression after `+`",
            Span::new(35, 36),
        )
        .with_context_label(Span::new(17, 24), "outputs block starts here")
        .with_secondary_label(Span::new(31, 33), "operator missing a right-hand side")
        .with_help("add a value after `+`");

        let rendered = diagnostic.render(src);
        assert!(
            rendered.contains("~ outputs block starts here"),
            "{}",
            rendered
        );
        assert!(
            rendered.contains("-- operator missing a right-hand side"),
            "{}",
            rendered
        );
        assert!(
            rendered.contains("= help: add a value after `+`"),
            "{}",
            rendered
        );
    }

    #[test]
    fn diagnostic_report_serializes_source_and_offsets() {
        let src = "dx(gut) = 1 +\n";
        let diagnostic = Diagnostic::error(
            DSL_PARSE_GENERIC,
            DiagnosticPhase::Parse,
            "expected expression after `+`",
            Span::new(12, 13),
        )
        .with_help("add a value after `+`");

        let report = DiagnosticReport::from_diagnostics("inline.dsl", Some(src), &[diagnostic]);
        let json = report.to_json().expect("serialize diagnostic report");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");

        assert_eq!(value["source"]["name"], "inline.dsl");
        assert_eq!(value["source"]["byte_len"], src.len());
        assert_eq!(value["diagnostics"][0]["code"], "DSL1000");
        assert_eq!(value["diagnostics"][0]["severity"], "error");
        assert_eq!(value["diagnostics"][0]["phase"], "parse");
        assert_eq!(value["diagnostics"][0]["labels"][0]["span"]["start"], 12);
        assert_eq!(
            value["diagnostics"][0]["labels"][0]["span"]["start_line"],
            1
        );
        assert_eq!(
            value["diagnostics"][0]["labels"][0]["span"]["start_column"],
            13
        );
        assert!(value["rendered"]
            .as_str()
            .expect("rendered output")
            .contains("error[DSL1000]"));
    }
}
