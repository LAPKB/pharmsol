use super::{ModelKind, Span};
use serde::{Deserialize, Serialize};

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
    pub kernel: AnalyticalKernel,
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
            "three_compartments_cl_with_absorption" => Some(Self::ThreeCompartmentsClWithAbsorption),
            "three_compartments_with_absorption" => Some(Self::ThreeCompartmentsWithAbsorption),
            _ => None,
        }
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