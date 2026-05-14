use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::name_match::{
    common_prefix_len, edit_distance, is_high_confidence_match, is_single_adjacent_transposition,
};
use crate::{ModelKind, RouteKind, Span};

pub type SymbolId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct TypedModule {
    pub models: Vec<TypedModel>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedModel {
    pub name: String,
    pub kind: ModelKind,
    pub symbols: Vec<Symbol>,
    pub parameters: Vec<SymbolId>,
    pub constants: Vec<TypedConstant>,
    pub covariates: Vec<TypedCovariate>,
    pub states: Vec<TypedState>,
    pub routes: Vec<TypedRoute>,
    pub derived: Vec<SymbolId>,
    pub outputs: Vec<SymbolId>,
    pub particles: Option<usize>,
    pub analytical: Option<TypedAnalytical>,
    pub derive: Option<TypedStatementBlock>,
    pub dynamics: Option<TypedStatementBlock>,
    pub outputs_block: TypedStatementBlock,
    pub init: Option<TypedStatementBlock>,
    pub drift: Option<TypedStatementBlock>,
    pub diffusion: Option<TypedStatementBlock>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub kind: SymbolKind,
    pub ty: SymbolType,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Parameter,
    Constant,
    Covariate,
    State,
    Route,
    Derived,
    Output,
    Local,
    LoopBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    Scalar(ValueType),
    Array { element: ValueType, size: usize },
    Route,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Int,
    Real,
    Bool,
}

impl ValueType {
    pub fn is_numeric(self) -> bool {
        matches!(self, ValueType::Int | ValueType::Real)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(i64),
    Real(f64),
    Bool(bool),
}

impl ConstValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            ConstValue::Int(_) => ValueType::Int,
            ConstValue::Real(_) => ValueType::Real,
            ConstValue::Bool(_) => ValueType::Bool,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConstValue::Int(value) => Some(*value as f64),
            ConstValue::Real(value) => Some(*value),
            ConstValue::Bool(_) => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConstValue::Int(value) => Some(*value),
            ConstValue::Real(value)
                if value.is_finite()
                    && value.fract() == 0.0
                    && *value >= i64::MIN as f64
                    && *value <= i64::MAX as f64 =>
            {
                Some(*value as i64)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedConstant {
    pub symbol: SymbolId,
    pub value: ConstValue,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovariateInterpolation {
    Linear,
    Locf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedCovariate {
    pub symbol: SymbolId,
    pub interpolation: Option<CovariateInterpolation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedState {
    pub symbol: SymbolId,
    pub size: Option<usize>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedRoute {
    pub symbol: SymbolId,
    pub kind: Option<RouteKind>,
    pub destination: TypedStatePlace,
    pub properties: Vec<TypedRouteProperty>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedRouteProperty {
    pub kind: RoutePropertyKind,
    pub value: TypedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RoutePropertyKind {
    Lag,
    Bioavailability,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedAnalytical {
    pub structure: AnalyticalKernel,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalyticalKernel {
    OneCompartment,
    OneCompartmentCl,
    OneCompartmentClWithAbsorption,
    OneCompartmentWithAbsorption,
    TwoCompartments,
    TwoCompartmentsCl,
    TwoCompartmentsClWithAbsorption,
    TwoCompartmentsWithAbsorption,
    ThreeCompartments,
    ThreeCompartmentsCl,
    ThreeCompartmentsClWithAbsorption,
    ThreeCompartmentsWithAbsorption,
}

impl AnalyticalKernel {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "one_compartment" => Some(Self::OneCompartment),
            "one_compartment_cl" => Some(Self::OneCompartmentCl),
            "one_compartment_cl_with_absorption" => Some(Self::OneCompartmentClWithAbsorption),
            "one_compartment_with_absorption" => Some(Self::OneCompartmentWithAbsorption),
            "two_compartments" => Some(Self::TwoCompartments),
            "two_compartments_cl" => Some(Self::TwoCompartmentsCl),
            "two_compartments_cl_with_absorption" => Some(Self::TwoCompartmentsClWithAbsorption),
            "two_compartments_with_absorption" => Some(Self::TwoCompartmentsWithAbsorption),
            "three_compartments" => Some(Self::ThreeCompartments),
            "three_compartments_cl" => Some(Self::ThreeCompartmentsCl),
            "three_compartments_cl_with_absorption" => {
                Some(Self::ThreeCompartmentsClWithAbsorption)
            }
            "three_compartments_with_absorption" => Some(Self::ThreeCompartmentsWithAbsorption),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::OneCompartment => "one_compartment",
            Self::OneCompartmentCl => "one_compartment_cl",
            Self::OneCompartmentClWithAbsorption => "one_compartment_cl_with_absorption",
            Self::OneCompartmentWithAbsorption => "one_compartment_with_absorption",
            Self::TwoCompartments => "two_compartments",
            Self::TwoCompartmentsCl => "two_compartments_cl",
            Self::TwoCompartmentsClWithAbsorption => "two_compartments_cl_with_absorption",
            Self::TwoCompartmentsWithAbsorption => "two_compartments_with_absorption",
            Self::ThreeCompartments => "three_compartments",
            Self::ThreeCompartmentsCl => "three_compartments_cl",
            Self::ThreeCompartmentsClWithAbsorption => "three_compartments_cl_with_absorption",
            Self::ThreeCompartmentsWithAbsorption => "three_compartments_with_absorption",
        }
    }

    pub fn required_parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::OneCompartment => &["ke"],
            Self::OneCompartmentCl => &["cl", "v"],
            Self::OneCompartmentClWithAbsorption => &["ka", "cl", "v"],
            Self::OneCompartmentWithAbsorption => &["ka", "ke"],
            Self::TwoCompartments => &["ke", "kcp", "kpc"],
            Self::TwoCompartmentsCl => &["cl", "q", "vc", "vp"],
            Self::TwoCompartmentsClWithAbsorption => &["ka", "cl", "q", "vc", "vp"],
            Self::TwoCompartmentsWithAbsorption => &["ke", "ka", "kcp", "kpc"],
            Self::ThreeCompartments => &["k10", "k12", "k13", "k21", "k31"],
            Self::ThreeCompartmentsCl => &["cl", "q2", "q3", "vc", "v2", "v3"],
            Self::ThreeCompartmentsClWithAbsorption => &["ka", "cl", "q2", "q3", "vc", "v2", "v3"],
            Self::ThreeCompartmentsWithAbsorption => &["ka", "k10", "k12", "k13", "k21", "k31"],
        }
    }

    pub fn required_parameter_count(self) -> usize {
        self.required_parameter_names().len()
    }

    pub fn state_count(self) -> usize {
        match self {
            Self::OneCompartment | Self::OneCompartmentCl => 1,
            Self::OneCompartmentClWithAbsorption | Self::OneCompartmentWithAbsorption => 2,
            Self::TwoCompartments | Self::TwoCompartmentsCl => 2,
            Self::TwoCompartmentsClWithAbsorption | Self::TwoCompartmentsWithAbsorption => 3,
            Self::ThreeCompartments | Self::ThreeCompartmentsCl => 3,
            Self::ThreeCompartmentsClWithAbsorption | Self::ThreeCompartmentsWithAbsorption => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticalStructureInputSource {
    Primary,
    Derived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyticalStructureInputBinding {
    pub source: AnalyticalStructureInputSource,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyticalStructureInputKind {
    AllPrimary {
        indices: Vec<usize>,
        identity: bool,
    },
    AllDerived {
        indices: Vec<usize>,
        identity: bool,
    },
    Mixed {
        bindings: Vec<AnalyticalStructureInputBinding>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalStructureInputPlan {
    kind: AnalyticalStructureInputKind,
}

impl AnalyticalStructureInputPlan {
    pub fn for_kernel<P, D>(
        kernel: AnalyticalKernel,
        primary_names: P,
        derived_names: D,
    ) -> Result<Self, AnalyticalStructureInputError>
    where
        P: IntoIterator,
        P::Item: AsRef<str>,
        D: IntoIterator,
        D::Item: AsRef<str>,
    {
        let primary_names = primary_names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();
        let derived_names = derived_names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();

        let mut primary_index_by_name =
            std::collections::HashMap::with_capacity(primary_names.len());
        for (index, name) in primary_names.iter().enumerate() {
            if primary_index_by_name.insert(name.as_str(), index).is_some() {
                return Err(AnalyticalStructureInputError::DuplicatePrimary { name: name.clone() });
            }
        }

        let mut derived_index_by_name =
            std::collections::HashMap::with_capacity(derived_names.len());
        for (index, name) in derived_names.iter().enumerate() {
            if derived_index_by_name.insert(name.as_str(), index).is_some() {
                return Err(AnalyticalStructureInputError::DuplicateDerived { name: name.clone() });
            }
            if primary_index_by_name.contains_key(name.as_str()) {
                return Err(AnalyticalStructureInputError::ConflictingName { name: name.clone() });
            }
        }

        let mut bindings = Vec::with_capacity(kernel.required_parameter_count());
        for required_name in kernel.required_parameter_names() {
            match (
                primary_index_by_name.get(required_name).copied(),
                derived_index_by_name.get(required_name).copied(),
            ) {
                (Some(index), None) => bindings.push(AnalyticalStructureInputBinding {
                    source: AnalyticalStructureInputSource::Primary,
                    index,
                }),
                (None, Some(index)) => bindings.push(AnalyticalStructureInputBinding {
                    source: AnalyticalStructureInputSource::Derived,
                    index,
                }),
                (None, None) => {
                    return Err(AnalyticalStructureInputError::MissingRequiredName {
                        structure: kernel.name(),
                        name: (*required_name).to_string(),
                        suggestion: best_similar_candidate(
                            required_name,
                            primary_names
                                .iter()
                                .chain(derived_names.iter())
                                .map(String::as_str),
                        ),
                    });
                }
                (Some(_), Some(_)) => {
                    return Err(AnalyticalStructureInputError::ConflictingName {
                        name: (*required_name).to_string(),
                    });
                }
            }
        }

        let all_primary = bindings
            .iter()
            .all(|binding| binding.source == AnalyticalStructureInputSource::Primary);
        if all_primary {
            let indices = bindings
                .iter()
                .map(|binding| binding.index)
                .collect::<Vec<_>>();
            let identity = indices
                .iter()
                .enumerate()
                .all(|(required_index, source_index)| required_index == *source_index);
            return Ok(Self {
                kind: AnalyticalStructureInputKind::AllPrimary { indices, identity },
            });
        }

        let all_derived = bindings
            .iter()
            .all(|binding| binding.source == AnalyticalStructureInputSource::Derived);
        if all_derived {
            let indices = bindings
                .iter()
                .map(|binding| binding.index)
                .collect::<Vec<_>>();
            let identity = indices
                .iter()
                .enumerate()
                .all(|(required_index, source_index)| required_index == *source_index);
            return Ok(Self {
                kind: AnalyticalStructureInputKind::AllDerived { indices, identity },
            });
        }

        Ok(Self {
            kind: AnalyticalStructureInputKind::Mixed { bindings },
        })
    }

    pub fn kind(&self) -> &AnalyticalStructureInputKind {
        &self.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalyticalStructureInputError {
    DuplicatePrimary {
        name: String,
    },
    DuplicateDerived {
        name: String,
    },
    ConflictingName {
        name: String,
    },
    MissingRequiredName {
        structure: &'static str,
        name: String,
        suggestion: Option<String>,
    },
}

impl fmt::Display for AnalyticalStructureInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePrimary { name } => write!(f, "duplicate primary parameter `{name}`"),
            Self::DuplicateDerived { name } => write!(f, "duplicate derived parameter `{name}`"),
            Self::ConflictingName { name } => {
                write!(f, "`{name}` is declared in both `params` and `derived`")
            }
            Self::MissingRequiredName {
                structure,
                name,
                suggestion,
            } => {
                write!(f, "analytical structure `{structure}` requires `{name}`; ")?;
                if let Some(candidate) = suggestion {
                    write!(f, "did you mean `{name}` instead of `{candidate}`? ")?;
                }
                f.write_str("declare it in `params` or `derived`")
            }
        }
    }
}

impl Error for AnalyticalStructureInputError {}

fn best_similar_candidate<'a, I>(needle: &str, candidates: I) -> Option<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let original_needle = needle;
    let needle = needle.to_ascii_lowercase();
    let mut best: Option<((usize, usize, usize), String)> = None;
    let mut tied = false;

    for candidate in candidates {
        if candidate == original_needle {
            continue;
        }

        let lookup = candidate.to_ascii_lowercase();
        let distance = if is_single_adjacent_transposition(&needle, &lookup) {
            1
        } else {
            edit_distance(&needle, &lookup)
        };
        let prefix = common_prefix_len(&needle, &lookup);
        if !is_high_confidence_match(&needle, &lookup, distance, prefix) {
            continue;
        }

        let score = (
            distance,
            usize::MAX - prefix,
            needle.len().abs_diff(lookup.len()),
        );
        match &best {
            None => {
                best = Some((score, candidate.to_string()));
                tied = false;
            }
            Some((best_score, _)) if score < *best_score => {
                best = Some((score, candidate.to_string()));
                tied = false;
            }
            Some((best_score, _)) if score == *best_score => tied = true,
            _ => {}
        }
    }

    if tied {
        None
    } else {
        best.map(|(_, candidate)| candidate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockContext {
    Derive,
    Dynamics,
    Outputs,
    Init,
    Drift,
    Diffusion,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedStatementBlock {
    pub context: BlockContext,
    pub statements: Vec<TypedStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedStmt {
    pub kind: TypedStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedStmtKind {
    Let(TypedLetStmt),
    Assign(TypedAssignStmt),
    If(TypedIfStmt),
    For(TypedForStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedLetStmt {
    pub symbol: SymbolId,
    pub value: TypedExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedAssignStmt {
    pub target: TypedAssignTarget,
    pub value: TypedExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedIfStmt {
    pub condition: TypedExpr,
    pub then_branch: Vec<TypedStmt>,
    pub else_branch: Option<Vec<TypedStmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedForStmt {
    pub binding: SymbolId,
    pub range: TypedRangeExpr,
    pub body: Vec<TypedStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedRangeExpr {
    pub start: TypedExpr,
    pub end: TypedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedAssignTarget {
    pub kind: TypedAssignTargetKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedAssignTargetKind {
    Derived(SymbolId),
    Output(SymbolId),
    StateInit(TypedStatePlace),
    Derivative(TypedStatePlace),
    Noise(TypedStatePlace),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedStatePlace {
    pub state: SymbolId,
    pub index: Option<Box<TypedExpr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    pub kind: TypedExprKind,
    pub ty: ValueType,
    pub constant: Option<ConstValue>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedExprKind {
    Literal(ConstValue),
    Symbol(SymbolId),
    StateValue(TypedStatePlace),
    Unary {
        op: TypedUnaryOp,
        expr: Box<TypedExpr>,
    },
    Binary {
        op: TypedBinaryOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
    },
    Call {
        callee: TypedCall,
        args: Vec<TypedExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedUnaryOp {
    Plus,
    Minus,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedBinaryOp {
    Or,
    And,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedCall {
    Math(MathIntrinsic),
    Rate(SymbolId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathIntrinsic {
    Abs,
    Ceil,
    Exp,
    Floor,
    Ln,
    Log,
    Log10,
    Log2,
    Max,
    Min,
    Pow,
    Round,
    Sin,
    Cos,
    Tan,
    Sqrt,
}

impl MathIntrinsic {
    pub const ALL: [Self; 16] = [
        Self::Abs,
        Self::Ceil,
        Self::Exp,
        Self::Floor,
        Self::Ln,
        Self::Log,
        Self::Log10,
        Self::Log2,
        Self::Max,
        Self::Min,
        Self::Pow,
        Self::Round,
        Self::Sin,
        Self::Cos,
        Self::Tan,
        Self::Sqrt,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Ceil => "ceil",
            Self::Exp => "exp",
            Self::Floor => "floor",
            Self::Ln => "ln",
            Self::Log => "log",
            Self::Log10 => "log10",
            Self::Log2 => "log2",
            Self::Max => "max",
            Self::Min => "min",
            Self::Pow => "pow",
            Self::Round => "round",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Sqrt => "sqrt",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "abs" => Some(Self::Abs),
            "ceil" => Some(Self::Ceil),
            "exp" => Some(Self::Exp),
            "floor" => Some(Self::Floor),
            "ln" => Some(Self::Ln),
            "log" => Some(Self::Log),
            "log10" => Some(Self::Log10),
            "log2" => Some(Self::Log2),
            "max" => Some(Self::Max),
            "min" => Some(Self::Min),
            "pow" => Some(Self::Pow),
            "round" => Some(Self::Round),
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "sqrt" => Some(Self::Sqrt),
            _ => None,
        }
    }

    pub fn arity(self) -> IntrinsicArity {
        match self {
            Self::Max | Self::Min | Self::Pow => IntrinsicArity::Exact(2),
            _ => IntrinsicArity::Exact(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicArity {
    Exact(usize),
}

#[cfg(test)]
mod tests {
    use super::{
        AnalyticalKernel, AnalyticalStructureInputBinding, AnalyticalStructureInputError,
        AnalyticalStructureInputKind, AnalyticalStructureInputPlan, AnalyticalStructureInputSource,
    };

    #[test]
    fn analytical_structures_publish_required_parameter_order() {
        let cases = [
            (AnalyticalKernel::OneCompartment, &["ke"][..]),
            (AnalyticalKernel::OneCompartmentCl, &["cl", "v"][..]),
            (
                AnalyticalKernel::OneCompartmentClWithAbsorption,
                &["ka", "cl", "v"][..],
            ),
            (
                AnalyticalKernel::OneCompartmentWithAbsorption,
                &["ka", "ke"][..],
            ),
            (AnalyticalKernel::TwoCompartments, &["ke", "kcp", "kpc"][..]),
            (
                AnalyticalKernel::TwoCompartmentsCl,
                &["cl", "q", "vc", "vp"][..],
            ),
            (
                AnalyticalKernel::TwoCompartmentsClWithAbsorption,
                &["ka", "cl", "q", "vc", "vp"][..],
            ),
            (
                AnalyticalKernel::TwoCompartmentsWithAbsorption,
                &["ke", "ka", "kcp", "kpc"][..],
            ),
            (
                AnalyticalKernel::ThreeCompartments,
                &["k10", "k12", "k13", "k21", "k31"][..],
            ),
            (
                AnalyticalKernel::ThreeCompartmentsCl,
                &["cl", "q2", "q3", "vc", "v2", "v3"][..],
            ),
            (
                AnalyticalKernel::ThreeCompartmentsClWithAbsorption,
                &["ka", "cl", "q2", "q3", "vc", "v2", "v3"][..],
            ),
            (
                AnalyticalKernel::ThreeCompartmentsWithAbsorption,
                &["ka", "k10", "k12", "k13", "k21", "k31"][..],
            ),
        ];

        for (kernel, expected) in cases {
            assert_eq!(kernel.required_parameter_names(), expected);
            assert_eq!(kernel.required_parameter_count(), expected.len());
            assert_eq!(AnalyticalKernel::from_name(kernel.name()), Some(kernel));
        }
    }

    #[test]
    fn builds_identity_all_primary_plan() {
        let plan = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentClWithAbsorption,
            ["ka", "cl", "v", "tvcl"],
            std::iter::empty::<&str>(),
        )
        .unwrap();

        assert_eq!(
            plan.kind(),
            &AnalyticalStructureInputKind::AllPrimary {
                indices: vec![0, 1, 2],
                identity: true,
            }
        );
    }

    #[test]
    fn builds_reordered_all_primary_plan() {
        let plan = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::TwoCompartmentsWithAbsorption,
            ["ka", "ke", "kcp", "kpc", "v"],
            std::iter::empty::<&str>(),
        )
        .unwrap();

        assert_eq!(
            plan.kind(),
            &AnalyticalStructureInputKind::AllPrimary {
                indices: vec![1, 0, 2, 3],
                identity: false,
            }
        );
    }

    #[test]
    fn builds_identity_all_derived_plan() {
        let plan = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka0", "ke0", "v"],
            ["ka", "ke"],
        )
        .unwrap();

        assert_eq!(
            plan.kind(),
            &AnalyticalStructureInputKind::AllDerived {
                indices: vec![0, 1],
                identity: true,
            }
        );
    }

    #[test]
    fn builds_mixed_source_plan() {
        let plan = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka", "ke0", "v"],
            ["ke"],
        )
        .unwrap();

        assert_eq!(
            plan.kind(),
            &AnalyticalStructureInputKind::Mixed {
                bindings: vec![
                    AnalyticalStructureInputBinding {
                        source: AnalyticalStructureInputSource::Primary,
                        index: 0,
                    },
                    AnalyticalStructureInputBinding {
                        source: AnalyticalStructureInputSource::Derived,
                        index: 0,
                    },
                ],
            }
        );
    }

    #[test]
    fn rejects_duplicate_primary_name() {
        let error = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka", "ka", "ke"],
            std::iter::empty::<&str>(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            AnalyticalStructureInputError::DuplicatePrimary {
                name: "ka".to_string(),
            }
        );
        assert_eq!(error.to_string(), "duplicate primary parameter `ka`");
    }

    #[test]
    fn rejects_duplicate_derived_name() {
        let error = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka", "ke0", "v"],
            ["ke", "ke"],
        )
        .unwrap_err();

        assert_eq!(
            error,
            AnalyticalStructureInputError::DuplicateDerived {
                name: "ke".to_string(),
            }
        );
        assert_eq!(error.to_string(), "duplicate derived parameter `ke`");
    }

    #[test]
    fn rejects_conflicting_primary_and_derived_name() {
        let error = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka", "ke", "v"],
            ["ke"],
        )
        .unwrap_err();

        assert_eq!(
            error,
            AnalyticalStructureInputError::ConflictingName {
                name: "ke".to_string(),
            }
        );
        assert_eq!(
            error.to_string(),
            "`ke` is declared in both `params` and `derived`"
        );
    }

    #[test]
    fn reports_missing_required_name_with_suggestion() {
        let error = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka", "kel", "v"],
            std::iter::empty::<&str>(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            AnalyticalStructureInputError::MissingRequiredName {
                structure: "one_compartment_with_absorption",
                name: "ke".to_string(),
                suggestion: Some("kel".to_string()),
            }
        );
        assert_eq!(
            error.to_string(),
            "analytical structure `one_compartment_with_absorption` requires `ke`; did you mean `ke` instead of `kel`? declare it in `params` or `derived`"
        );
    }

    #[test]
    fn reports_missing_required_name_without_suggestion() {
        let error = AnalyticalStructureInputPlan::for_kernel(
            AnalyticalKernel::OneCompartmentWithAbsorption,
            ["ka0", "v"],
            std::iter::empty::<&str>(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            AnalyticalStructureInputError::MissingRequiredName {
                structure: "one_compartment_with_absorption",
                name: "ka".to_string(),
                suggestion: None,
            }
        );
        assert_eq!(
            error.to_string(),
            "analytical structure `one_compartment_with_absorption` requires `ka`; declare it in `params` or `derived`"
        );
    }
}
