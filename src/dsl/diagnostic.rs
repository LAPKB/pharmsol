use std::fmt;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn render(&self, src: &str) -> String {
        let offset = self.span.start.min(src.len());
        let (line, column, line_start, line_end) = line_info(src, offset);
        let line_text = &src[line_start..line_end];
        let caret_start = src[line_start..offset].chars().count();
        let highlight_end = self.span.end.min(line_end).max(offset);
        let caret_len = src[offset..highlight_end].chars().count().max(1);
        let gutter = line.to_string().len();

        let mut rendered = String::new();
        rendered.push_str(&format!(
            "line {}, column {}: {}\n",
            line, column, self.message
        ));
        rendered.push_str(&format!(
            "{:>width$} | {}\n",
            line,
            line_text,
            width = gutter
        ));
        rendered.push_str(&format!(
            "{:>width$} | {}{}\n",
            "",
            " ".repeat(caret_start),
            "^".repeat(caret_len),
            width = gutter
        ));
        for note in &self.notes {
            rendered.push_str(&format!(
                "{:>width$} = note: {}\n",
                "",
                note,
                width = gutter
            ));
        }
        rendered
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    diagnostic: Diagnostic,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            diagnostic: Diagnostic::new(message, span),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.diagnostic.notes.push(note.into());
        self
    }

    pub fn diagnostic(&self) -> &Diagnostic {
        &self.diagnostic
    }

    pub fn render(&self, src: &str) -> String {
        self.diagnostic.render(src)
    }

    pub fn shifted(mut self, delta: usize) -> Self {
        self.diagnostic.span = self.diagnostic.span.shifted(delta);
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at bytes {}..{}",
            self.diagnostic.message, self.diagnostic.span.start, self.diagnostic.span.end
        )
    }
}

impl std::error::Error for ParseError {}

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
