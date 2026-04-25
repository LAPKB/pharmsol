use std::fmt::{self, Write};

use serde::{Deserialize, Serialize};

use super::diagnostic::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub models: Vec<Model>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: Ident,
    pub kind: ModelKind,
    pub items: Vec<ModelItem>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub text: String,
    pub span: Span,
}

impl Ident {
    pub fn new(text: impl Into<String>, span: Span) -> Self {
        Self {
            text: text.into(),
            span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    Ode,
    Analytical,
    Sde,
}

impl ModelKind {
    pub fn keyword(self) -> &'static str {
        match self {
            ModelKind::Ode => "ode",
            ModelKind::Analytical => "analytical",
            ModelKind::Sde => "sde",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelItem {
    Parameters(ParametersBlock),
    Constants(ConstantsBlock),
    Covariates(CovariatesBlock),
    States(StatesBlock),
    Routes(RoutesBlock),
    Derive(StatementBlock),
    Dynamics(StatementBlock),
    Outputs(StatementBlock),
    Analytical(AnalyticalBlock),
    Init(StatementBlock),
    Drift(StatementBlock),
    Diffusion(StatementBlock),
    Particles(ParticlesDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParametersBlock {
    pub items: Vec<Ident>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantsBlock {
    pub items: Vec<Binding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CovariatesBlock {
    pub items: Vec<CovariateDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CovariateDecl {
    pub name: Ident,
    pub interpolation: Option<Ident>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatesBlock {
    pub items: Vec<StateDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateDecl {
    pub name: Ident,
    pub size: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutesBlock {
    pub routes: Vec<RouteDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecl {
    pub input: Ident,
    pub destination: Place,
    pub properties: Vec<Binding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub name: Ident,
    pub index: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatementBlock {
    pub statements: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticalBlock {
    pub kernel: Ident,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParticlesDecl {
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Let(LetStmt),
    Assign(AssignStmt),
    If(IfStmt),
    For(ForStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt {
    pub name: Ident,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignStmt {
    pub target: AssignTarget,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_branch: Vec<Stmt>,
    pub else_branch: Option<Vec<Stmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub binding: Ident,
    pub range: RangeExpr,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeExpr {
    pub start: Expr,
    pub end: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignTarget {
    pub kind: AssignTargetKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTargetKind {
    Name(Ident),
    Index { target: Ident, index: Expr },
    Call { callee: Ident, args: Vec<Expr> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Number(f64),
    Bool(bool),
    Name(Ident),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        callee: Ident,
        args: Vec<Expr>,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
}

impl UnaryOp {
    fn symbol(self) -> &'static str {
        match self {
            UnaryOp::Plus => "+",
            UnaryOp::Minus => "-",
            UnaryOp::Not => "!",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
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

impl BinaryOp {
    fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Or => "||",
            BinaryOp::And => "&&",
            BinaryOp::Eq => "==",
            BinaryOp::NotEq => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Pow => "^",
        }
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        for (index, model) in self.models.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            write_model(model, &mut out, 0)?;
            out.push('\n');
        }
        f.write_str(out.trim_end_matches('\n'))
    }
}

fn write_model(model: &Model, out: &mut String, indent_level: usize) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "model {} {{", model.name.text)?;
    indent(out, indent_level + 1);
    writeln!(out, "kind {}", model.kind.keyword())?;
    for item in &model.items {
        out.push('\n');
        write_item(item, out, indent_level + 1)?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_item(item: &ModelItem, out: &mut String, indent_level: usize) -> fmt::Result {
    match item {
        ModelItem::Parameters(block) => {
            write_ident_block("parameters", &block.items, out, indent_level)
        }
        ModelItem::Constants(block) => {
            write_binding_block("constants", &block.items, out, indent_level)
        }
        ModelItem::Covariates(block) => write_covariates_block(block, out, indent_level),
        ModelItem::States(block) => write_states_block(block, out, indent_level),
        ModelItem::Routes(block) => write_routes_block(block, out, indent_level),
        ModelItem::Derive(block) => {
            write_statement_block("derive", &block.statements, out, indent_level)
        }
        ModelItem::Dynamics(block) => {
            write_statement_block("dynamics", &block.statements, out, indent_level)
        }
        ModelItem::Outputs(block) => {
            write_statement_block("outputs", &block.statements, out, indent_level)
        }
        ModelItem::Analytical(block) => write_analytical_block(block, out, indent_level),
        ModelItem::Init(block) => {
            write_statement_block("init", &block.statements, out, indent_level)
        }
        ModelItem::Drift(block) => {
            write_statement_block("drift", &block.statements, out, indent_level)
        }
        ModelItem::Diffusion(block) => {
            write_statement_block("diffusion", &block.statements, out, indent_level)
        }
        ModelItem::Particles(decl) => {
            indent(out, indent_level);
            write!(out, "particles ")?;
            write_expr(&decl.value, out)?;
            Ok(())
        }
    }
}

fn write_ident_block(
    name: &str,
    items: &[Ident],
    out: &mut String,
    indent_level: usize,
) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "{} {{", name)?;
    for ident in items {
        indent(out, indent_level + 1);
        writeln!(out, "{},", ident.text)?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_binding_block(
    name: &str,
    items: &[Binding],
    out: &mut String,
    indent_level: usize,
) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "{} {{", name)?;
    for item in items {
        indent(out, indent_level + 1);
        write!(out, "{} = ", item.name.text)?;
        write_expr(&item.value, out)?;
        writeln!(out)?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_covariates_block(
    block: &CovariatesBlock,
    out: &mut String,
    indent_level: usize,
) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "covariates {{")?;
    for covariate in &block.items {
        indent(out, indent_level + 1);
        write!(out, "{}", covariate.name.text)?;
        if let Some(annotation) = &covariate.interpolation {
            write!(out, " @{}", annotation.text)?;
        }
        writeln!(out, ",")?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_states_block(block: &StatesBlock, out: &mut String, indent_level: usize) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "states {{")?;
    for state in &block.items {
        indent(out, indent_level + 1);
        write!(out, "{}", state.name.text)?;
        if let Some(size) = &state.size {
            out.push('[');
            write_expr(size, out)?;
            out.push(']');
        }
        writeln!(out, ",")?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_routes_block(block: &RoutesBlock, out: &mut String, indent_level: usize) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "routes {{")?;
    for (index, route) in block.routes.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        indent(out, indent_level + 1);
        write!(
            out,
            "{} -> {}",
            route.input.text, route.destination.name.text
        )?;
        if let Some(index_expr) = &route.destination.index {
            out.push('[');
            write_expr(index_expr, out)?;
            out.push(']');
        }
        if route.properties.is_empty() {
            writeln!(out)?;
            continue;
        }

        writeln!(out, " {{")?;
        for property in &route.properties {
            indent(out, indent_level + 2);
            write!(out, "{} = ", property.name.text)?;
            write_expr(&property.value, out)?;
            writeln!(out)?;
        }
        indent(out, indent_level + 1);
        writeln!(out, "}}")?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_statement_block(
    name: &str,
    statements: &[Stmt],
    out: &mut String,
    indent_level: usize,
) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "{} {{", name)?;
    for stmt in statements {
        write_stmt(stmt, out, indent_level + 1)?;
    }
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_analytical_block(
    block: &AnalyticalBlock,
    out: &mut String,
    indent_level: usize,
) -> fmt::Result {
    indent(out, indent_level);
    writeln!(out, "analytical {{")?;
    indent(out, indent_level + 1);
    writeln!(out, "kernel = {}", block.kernel.text)?;
    indent(out, indent_level);
    write!(out, "}}")
}

fn write_stmt(stmt: &Stmt, out: &mut String, indent_level: usize) -> fmt::Result {
    match &stmt.kind {
        StmtKind::Let(let_stmt) => {
            indent(out, indent_level);
            write!(out, "let {} = ", let_stmt.name.text)?;
            write_expr(&let_stmt.value, out)?;
            writeln!(out)
        }
        StmtKind::Assign(assign) => {
            indent(out, indent_level);
            write_assign_target(&assign.target, out)?;
            write!(out, " = ")?;
            write_expr(&assign.value, out)?;
            writeln!(out)
        }
        StmtKind::If(if_stmt) => {
            indent(out, indent_level);
            write!(out, "if ")?;
            write_expr(&if_stmt.condition, out)?;
            writeln!(out, " {{")?;
            for stmt in &if_stmt.then_branch {
                write_stmt(stmt, out, indent_level + 1)?;
            }
            indent(out, indent_level);
            if let Some(else_branch) = &if_stmt.else_branch {
                writeln!(out, "}} else {{")?;
                for stmt in else_branch {
                    write_stmt(stmt, out, indent_level + 1)?;
                }
                indent(out, indent_level);
                writeln!(out, "}}")
            } else {
                writeln!(out, "}}")
            }
        }
        StmtKind::For(for_stmt) => {
            indent(out, indent_level);
            write!(out, "for {} in ", for_stmt.binding.text)?;
            write_expr(&for_stmt.range.start, out)?;
            write!(out, "..")?;
            write_expr(&for_stmt.range.end, out)?;
            writeln!(out, " {{")?;
            for stmt in &for_stmt.body {
                write_stmt(stmt, out, indent_level + 1)?;
            }
            indent(out, indent_level);
            writeln!(out, "}}")
        }
    }
}

fn write_assign_target(target: &AssignTarget, out: &mut String) -> fmt::Result {
    match &target.kind {
        AssignTargetKind::Name(name) => out.write_str(&name.text),
        AssignTargetKind::Index { target, index } => {
            write!(out, "{}[", target.text)?;
            write_expr(index, out)?;
            out.push(']');
            Ok(())
        }
        AssignTargetKind::Call { callee, args } => {
            write!(out, "{}(", callee.text)?;
            write_expr_list(args, out)?;
            out.push(')');
            Ok(())
        }
    }
}

fn write_expr(expr: &Expr, out: &mut String) -> fmt::Result {
    match &expr.kind {
        ExprKind::Number(value) => write!(out, "{}", value),
        ExprKind::Bool(value) => write!(out, "{}", value),
        ExprKind::Name(name) => out.write_str(&name.text),
        ExprKind::Unary { op, expr } => {
            write!(out, "{}", op.symbol())?;
            write_wrapped_expr(expr, out)
        }
        ExprKind::Binary { op, lhs, rhs } => {
            out.push('(');
            write_expr(lhs, out)?;
            write!(out, " {} ", op.symbol())?;
            write_expr(rhs, out)?;
            out.push(')');
            Ok(())
        }
        ExprKind::Call { callee, args } => {
            write!(out, "{}(", callee.text)?;
            write_expr_list(args, out)?;
            out.push(')');
            Ok(())
        }
        ExprKind::Index { target, index } => {
            write_wrapped_expr(target, out)?;
            out.push('[');
            write_expr(index, out)?;
            out.push(']');
            Ok(())
        }
    }
}

fn write_wrapped_expr(expr: &Expr, out: &mut String) -> fmt::Result {
    match expr.kind {
        ExprKind::Number(_)
        | ExprKind::Bool(_)
        | ExprKind::Name(_)
        | ExprKind::Call { .. }
        | ExprKind::Index { .. } => write_expr(expr, out),
        _ => {
            out.push('(');
            write_expr(expr, out)?;
            out.push(')');
            Ok(())
        }
    }
}

fn write_expr_list(args: &[Expr], out: &mut String) -> fmt::Result {
    for (index, arg) in args.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write_expr(arg, out)?;
    }
    Ok(())
}

fn indent(out: &mut String, level: usize) {
    out.push_str(&"  ".repeat(level));
}
