//! Plain-text model format for runtime authoring.
//!
//! Lets a downstream user (e.g. an R or Python wrapper, or anyone with only
//! the precompiled `pharmsol` library) define a complete JIT model from a
//! string and run it without invoking any Rust toolchain.
//!
//! # Format
//!
//! The format is line-oriented. Blank lines and lines starting with `#` are
//! ignored. Each non-empty line is one of:
//!
//! ```text
//! name        = my_model
//! compartments = depot, central
//! params      = ka, CL, V
//! covariates  = WT
//! ndrugs      = 1
//! dxdt(depot)   = -ka * depot
//! dxdt(central) =  ka * depot - (CL * pow(WT/70.0, 0.75) / V) * central
//! out(cp)       =  central / V
//! ```
//!
//! Order of declarations is flexible, but every compartment listed in
//! `compartments` must have a matching `dxdt(...)` line. `out(...)` lines
//! define output equations in the order they appear (output `0` is the first
//! `out` line, etc.).
//!
//! Lists are comma-separated; whitespace is ignored. The expression on the
//! right of `=` follows the syntax documented in [`crate::jit`].
//!
//! # Example
//!
//! ```ignore
//! use pharmsol::jit::Model;
//! let src = r#"
//!     name = onecmt
//!     compartments = central
//!     params = CL, V
//!     dxdt(central) = rateiv[0] - (CL / V) * central
//!     out(cp) = central / V
//! "#;
//! let ode = Model::from_text(src)?.compile()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use thiserror::Error;

use super::model::Model;

#[derive(Debug, Error)]
pub enum TextError {
    #[error("line {line}: expected `key = value` or `dxdt(name) = expr` or `out(name) = expr`")]
    BadLine { line: usize },
    #[error("line {line}: unknown key {key:?} (expected name, compartments, params, covariates, ndrugs, dxdt(...), out(...), init(...), lag(N), fa(N), let NAME)")]
    UnknownKey { line: usize, key: String },
    #[error("line {line}: malformed `{kind}(...)` header: {raw:?}")]
    BadHeader {
        line: usize,
        kind: &'static str,
        raw: String,
    },
    #[error("line {line}: invalid integer for ndrugs: {value:?}")]
    BadNdrugs { line: usize, value: String },
    #[error("line {line}: invalid integer for {kind} channel: {value:?}")]
    BadChannel {
        line: usize,
        kind: &'static str,
        value: String,
    },
    #[error("line {line}: malformed `let` declaration: {raw:?}")]
    BadLet { line: usize, raw: String },
    #[error("model has no `compartments` declaration")]
    NoCompartments,
}

/// Parse a model from the text format described in [`crate::jit::text`].
pub(crate) fn parse_text(src: &str) -> Result<Model, TextError> {
    let mut name = String::from("model");
    let mut compartments: Option<Vec<String>> = None;
    let mut params: Vec<String> = Vec::new();
    let mut covariates: Vec<String> = Vec::new();
    let mut ndrugs: Option<usize> = None;
    let mut dxdt: Vec<(String, String)> = Vec::new();
    let mut outs: Vec<(String, String)> = Vec::new();
    let mut inits: Vec<(String, String)> = Vec::new();
    let mut lags: Vec<(usize, String)> = Vec::new();
    let mut fas: Vec<(usize, String)> = Vec::new();
    let mut lets: Vec<(String, String)> = Vec::new();

    for (lineno, raw_line) in src.lines().enumerate() {
        let line_num = lineno + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let Some(eq_idx) = line.find('=') else {
            return Err(TextError::BadLine { line: line_num });
        };
        let lhs = line[..eq_idx].trim();
        let rhs = line[eq_idx + 1..].trim();

        // `let NAME = expr`
        if let Some(rest) = lhs.strip_prefix("let ") {
            let name_tok = rest.trim();
            if name_tok.is_empty()
                || !name_tok
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                || name_tok
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(true)
            {
                return Err(TextError::BadLet {
                    line: line_num,
                    raw: lhs.to_string(),
                });
            }
            lets.push((name_tok.to_string(), rhs.to_string()));
            continue;
        }

        if let Some(target) = paren_target(lhs, "dxdt") {
            dxdt.push((
                target.ok_or_else(|| TextError::BadHeader {
                    line: line_num,
                    kind: "dxdt",
                    raw: lhs.to_string(),
                })?,
                rhs.to_string(),
            ));
            continue;
        }
        if let Some(target) = paren_target(lhs, "out") {
            outs.push((
                target.ok_or_else(|| TextError::BadHeader {
                    line: line_num,
                    kind: "out",
                    raw: lhs.to_string(),
                })?,
                rhs.to_string(),
            ));
            continue;
        }
        if let Some(target) = paren_target(lhs, "init") {
            inits.push((
                target.ok_or_else(|| TextError::BadHeader {
                    line: line_num,
                    kind: "init",
                    raw: lhs.to_string(),
                })?,
                rhs.to_string(),
            ));
            continue;
        }
        if let Some(target) = paren_target(lhs, "lag") {
            let s = target.ok_or_else(|| TextError::BadHeader {
                line: line_num,
                kind: "lag",
                raw: lhs.to_string(),
            })?;
            let ch: usize = s.parse().map_err(|_| TextError::BadChannel {
                line: line_num,
                kind: "lag",
                value: s.clone(),
            })?;
            lags.push((ch, rhs.to_string()));
            continue;
        }
        if let Some(target) = paren_target(lhs, "fa") {
            let s = target.ok_or_else(|| TextError::BadHeader {
                line: line_num,
                kind: "fa",
                raw: lhs.to_string(),
            })?;
            let ch: usize = s.parse().map_err(|_| TextError::BadChannel {
                line: line_num,
                kind: "fa",
                value: s.clone(),
            })?;
            fas.push((ch, rhs.to_string()));
            continue;
        }

        match lhs {
            "name" => name = rhs.to_string(),
            "compartments" => compartments = Some(split_list(rhs)),
            "params" | "parameters" => params = split_list(rhs),
            "covariates" | "covs" => covariates = split_list(rhs),
            "ndrugs" => {
                ndrugs = Some(rhs.parse().map_err(|_| TextError::BadNdrugs {
                    line: line_num,
                    value: rhs.to_string(),
                })?);
            }
            other => {
                return Err(TextError::UnknownKey {
                    line: line_num,
                    key: other.to_string(),
                });
            }
        }
    }

    let compartments = compartments.ok_or(TextError::NoCompartments)?;

    let mut m = Model::new(name)
        .compartments(compartments)
        .params(params)
        .covariates(covariates);
    if let Some(n) = ndrugs {
        m = m.ndrugs(n);
    }
    for (name, expr) in lets {
        m = m.let_binding(name, expr);
    }
    for (target, expr) in dxdt {
        m = m.dxdt(target, expr);
    }
    for (target, expr) in outs {
        m = m.output(target, expr);
    }
    for (target, expr) in inits {
        m = m.init(target, expr);
    }
    for (ch, expr) in lags {
        m = m.lag(ch, expr);
    }
    for (ch, expr) in fas {
        m = m.fa(ch, expr);
    }
    Ok(m)
}

fn strip_comment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn split_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// If `lhs` looks like `kind(target)`, return `Some(Some(target))`. If it
/// starts with `kind(` but is malformed, return `Some(None)`. Otherwise
/// `None`, meaning the line isn't of this kind.
fn paren_target(lhs: &str, kind: &str) -> Option<Option<String>> {
    let prefix_open = format!("{kind}(");
    if !lhs.starts_with(&prefix_open) {
        return None;
    }
    let inside = &lhs[prefix_open.len()..];
    let Some(close) = inside.rfind(')') else {
        return Some(None);
    };
    let target = inside[..close].trim().to_string();
    let trailing = inside[close + 1..].trim();
    if target.is_empty() || !trailing.is_empty() {
        return Some(None);
    }
    Some(Some(target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_model() {
        let src = r#"
            # one-compartment IV bolus
            name = onecmt
            compartments = central
            params = CL, V
            dxdt(central) = rateiv[0] - (CL / V) * central
            out(cp) = central / V
        "#;
        let m = parse_text(src).expect("parse");
        assert_eq!(m.name, "onecmt");
        assert_eq!(m.compartments, vec!["central".to_string()]);
        assert_eq!(m.params.len(), 2);
        assert_eq!(m.dxdt.len(), 1);
        assert_eq!(m.outputs.len(), 1);
    }

    #[test]
    fn rejects_unknown_key() {
        let err = parse_text("foo = bar").unwrap_err();
        assert!(matches!(err, TextError::UnknownKey { .. }));
    }
}
