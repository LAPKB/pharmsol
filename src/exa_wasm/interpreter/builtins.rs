//! Builtin function metadata used by the interpreter and typechecker.
use std::ops::RangeInclusive;

/// Return true if the name is a known builtin function.
pub fn is_known_function(name: &str) -> bool {
    match name {
        "exp" | "if" | "ln" | "log" | "log10" | "log2" | "sqrt" | "pow" | "powf" | "min"
        | "max" | "abs" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" => true,
        _ => false,
    }
}

/// Return the allowed argument count range for a builtin, if known.
/// Use inclusive ranges; None means unknown function.
pub fn arg_count_range(name: &str) -> Option<RangeInclusive<usize>> {
    match name {
        "exp" | "ln" | "log" | "log10" | "log2" | "sqrt" | "abs" | "floor" | "ceil" | "round"
        | "sin" | "cos" | "tan" => Some(1..=1),
        "pow" | "powf" | "min" | "max" => Some(2..=2),
        "if" => Some(3..=3),
        _ => None,
    }
}
