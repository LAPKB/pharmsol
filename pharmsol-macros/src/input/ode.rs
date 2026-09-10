//! `ode!` input.

use proc_macro2::{Span, TokenStream as TokenStream2};
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream},
    visit::{self, Visit},
    Expr, ExprClosure, Ident, LitStr, Token,
};

use super::{missing_required_field, parse_ident_list, parse_symbolic_index_list, set_once};
use crate::analysis::closure_param_ident;
use crate::crate_path::{parse_crate_marker, resolve_crate_path};
use crate::symbols::{symbolic_index_idents, OdeRouteDecl, OdeRouteKind, SymbolicIndex};
use crate::validate::{
    validate_named_binding_compatibility, validate_routes, validate_unique_idents,
    validate_unique_symbolic_indices, CommonBindingClosures, NamedBindingSets, OdeBindingClosures,
};

const MACRO_LABEL: &str = "declaration-first `ode!`";

pub(crate) struct OdeInput {
    pub(crate) name: LitStr,
    pub(crate) krate: TokenStream2,
    pub(crate) params: Vec<Ident>,
    pub(crate) covariates: Vec<Ident>,
    pub(crate) states: Vec<Ident>,
    pub(crate) outputs: Vec<SymbolicIndex>,
    pub(crate) routes: Vec<OdeRouteDecl>,
    pub(crate) diffeq: ExprClosure,
    pub(crate) lag: Option<ExprClosure>,
    pub(crate) init: Option<ExprClosure>,
    pub(crate) out: ExprClosure,
}

impl Parse for OdeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forwarded_krate = parse_crate_marker(input)?;
        let mut name = None;
        let mut krate = None;
        let mut params = None;
        let mut covariates = None;
        let mut states = None;
        let mut outputs = None;
        let mut diffeq = None;
        let mut lag = None;
        let mut init = None;
        let mut out = None;

        while !input.is_empty() {
            let key: Ident = input.call(Ident::parse_any)?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "name" => set_once(&mut name, input.parse()?, &key, "name", "ode!")?,
                "crate" => set_once(&mut krate, input.parse::<LitStr>()?, &key, "crate", "ode!")?,
                "params" => set_once(
                    &mut params,
                    parse_ident_list(input)?,
                    &key,
                    "params",
                    "ode!",
                )?,
                "covariates" => set_once(
                    &mut covariates,
                    parse_ident_list(input)?,
                    &key,
                    "covariates",
                    "ode!",
                )?,
                "states" => set_once(
                    &mut states,
                    parse_ident_list(input)?,
                    &key,
                    "states",
                    "ode!",
                )?,
                "outputs" => set_once(
                    &mut outputs,
                    parse_symbolic_index_list(input)?,
                    &key,
                    "outputs",
                    "ode!",
                )?,
                "routes" | "fa" => return Err(syn::Error::new_spanned(
                    &key,
                    "`ode!` inputs now belong in `diffeq`: use `bolus[input] * scale` or `infusion[input] * scale`; `routes` and `fa` fields have been removed",
                )),
                "diffeq" => set_once(&mut diffeq, input.parse()?, &key, "diffeq", "ode!")?,
                "lag" => set_once(&mut lag, input.parse()?, &key, "lag", "ode!")?,
                "init" => set_once(&mut init, input.parse()?, &key, "init", "ode!")?,
                "out" => set_once(&mut out, input.parse()?, &key, "out", "ode!")?,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!(
                            "unknown field `{other}`, expected one of: name, crate, params, covariates, states, outputs, diffeq, lag, init, out"
                        ),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "declaration-first `ode!` requires `name`, `params`, `states`, and `outputs`; the old inferred-dimensions form has been removed",
            )
        })?;
        let krate = resolve_crate_path(krate, forwarded_krate)?;
        let params = params.ok_or_else(|| missing_required_field("params", MACRO_LABEL))?;
        let covariates = covariates.unwrap_or_default();
        let states = states.ok_or_else(|| missing_required_field("states", MACRO_LABEL))?;
        let outputs = outputs.ok_or_else(|| missing_required_field("outputs", MACRO_LABEL))?;
        let diffeq = diffeq.ok_or_else(|| missing_required_field("diffeq", MACRO_LABEL))?;
        let out = out.ok_or_else(|| missing_required_field("out", MACRO_LABEL))?;
        let routes = infer_ode_routes(&diffeq)?;

        validate_unique_idents("parameter", &params, "ode!")?;
        validate_unique_idents("covariate", &covariates, "ode!")?;
        validate_unique_idents("state", &states, "ode!")?;
        let output_idents = symbolic_index_idents(&outputs);

        validate_unique_symbolic_indices("output", &outputs, "ode!")?;
        validate_routes(&routes, &states, "ode!")?;
        validate_named_binding_compatibility(
            NamedBindingSets {
                params: &params,
                derived: &[],
                covariates: &covariates,
                states: &states,
                outputs: &output_idents,
                routes: &routes,
            },
            OdeBindingClosures {
                diffeq: &diffeq,
                common: CommonBindingClosures {
                    lag: lag.as_ref(),
                    fa: None,
                    init: init.as_ref(),
                    out: &out,
                },
            },
        )?;

        Ok(Self {
            name,
            krate,
            params,
            covariates,
            states,
            outputs,
            routes,
            diffeq,
            lag,
            init,
            out,
        })
    }
}

/// Recognize the public RHS input forms, never arbitrary Rust vector indexing.
pub(crate) fn ode_rhs_input(expr: &Expr) -> syn::Result<Option<(OdeRouteKind, SymbolicIndex)>> {
    let (function, argument) = match expr {
        Expr::Index(index) => (index.expr.as_ref(), index.index.as_ref()),
        Expr::Call(call) if call.args.len() == 1 => (call.func.as_ref(), &call.args[0]),
        _ => return Ok(None),
    };
    let Expr::Path(path) = function else {
        return Ok(None);
    };
    let Some(name) = path.path.get_ident() else {
        return Ok(None);
    };
    let kind = match name.to_string().as_str() {
        "bolus" => OdeRouteKind::Bolus,
        "infusion" => OdeRouteKind::Infusion,
        _ => return Ok(None),
    };
    let input = syn::parse2::<SymbolicIndex>(quote::quote!(#argument)).map_err(|_| {
        syn::Error::new_spanned(argument, "an ODE input must be a single input label")
    })?;
    Ok(Some((kind, input)))
}

fn infer_ode_routes(diffeq: &ExprClosure) -> syn::Result<Vec<OdeRouteDecl>> {
    let dx_index = match diffeq.inputs.len() {
        3 => 2,
        5 => 3,
        _ => return Err(syn::Error::new_spanned(diffeq,
            "`ode!` requires `diffeq` to have either 5 parameters: |x, p, t, dx, cov| or 3 parameters: |x, t, dx|")),
    };
    let dx = closure_param_ident(diffeq, dx_index).ok_or_else(|| {
        syn::Error::new_spanned(diffeq, "the derivative parameter must have a name")
    })?;
    let mut visitor = OdeInputVisitor {
        dx,
        destination: None,
        routes: Vec::new(),
        error: None,
    };
    visitor.visit_expr(&diffeq.body);
    match visitor.error {
        Some(error) => Err(error),
        None => Ok(visitor.routes),
    }
}

struct OdeInputVisitor {
    dx: Ident,
    destination: Option<Ident>,
    routes: Vec<OdeRouteDecl>,
    error: Option<syn::Error>,
}

impl OdeInputVisitor {
    fn destination(&self, expr: &Expr) -> Option<Ident> {
        let Expr::Index(index) = expr else {
            return None;
        };
        let Expr::Path(base) = index.expr.as_ref() else {
            return None;
        };
        if !base.path.is_ident(&self.dx) {
            return None;
        }
        let Expr::Path(state) = index.index.as_ref() else {
            return None;
        };
        state.path.get_ident().cloned()
    }

    fn record(&mut self, kind: OdeRouteKind, input: SymbolicIndex, expr: &Expr) -> syn::Result<()> {
        let destination = self.destination.as_ref().ok_or_else(|| syn::Error::new_spanned(
            expr, "ODE inputs must appear in the RHS of a named derivative assignment such as `dx[central] = infusion[iv] * scale`",
        ))?;
        if let Some(existing) = self
            .routes
            .iter()
            .find(|route| route.kind == kind && route.input.name() == input.name())
        {
            if kind == OdeRouteKind::Bolus && existing.destination != *destination {
                return Err(syn::Error::new_spanned(
                    expr,
                    "an ODE bolus input must have one destination state",
                ));
            }
        } else {
            self.routes.push(OdeRouteDecl {
                kind,
                input,
                destination: destination.clone(),
            });
        }
        Ok(())
    }
}

impl<'ast> Visit<'ast> for OdeInputVisitor {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if self.error.is_some() {
            return;
        }
        match ode_rhs_input(expr) {
            Ok(Some((kind, input))) => {
                if let Err(error) = self.record(kind, input, expr) {
                    self.error = Some(error);
                }
                return;
            }
            Err(error) => {
                self.error = Some(error);
                return;
            }
            Ok(None) => {}
        }
        if let Expr::Assign(assign) = expr {
            let previous = self.destination.take();
            self.destination = self.destination(&assign.left);
            self.visit_expr(&assign.right);
            self.destination = previous;
        } else {
            visit::visit_expr(self, expr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::{dense_index_len, ode_route_input_bindings};

    fn source(equations: &str) -> String {
        format!("name: \"demo\", params: [ke], states: [depot, central], outputs: [cp], diffeq: |x, t, dx| {{ {equations} }}, out: |x, t, y| {{ y[cp] = x[central]; }}")
    }

    #[test]
    fn crate_sources_keep_their_precedence() {
        for (prefix, expected) in [
            ("crate: \"pmcore::pharmsol\", ", ":: pmcore :: pharmsol"),
            (
                "@pharmsol_crate(::reexporter::pharmsol) ",
                ":: reexporter :: pharmsol",
            ),
            (
                "@pharmsol_crate(::reexporter::pharmsol) crate: \"my_vendor::pharmsol\", ",
                ":: my_vendor :: pharmsol",
            ),
        ] {
            let input = syn::parse_str::<OdeInput>(&format!(
                "{prefix}{}",
                source("dx[central] = infusion[iv] - ke*x[central];")
            ))
            .unwrap();
            assert_eq!(input.krate.to_string(), expected);
        }
        assert!(syn::parse_str::<OdeInput>(&format!(
            "crate: \"pmcore::pharmsol<T>\", {}",
            source("")
        ))
        .err()
        .unwrap()
        .to_string()
        .contains("without generic arguments"));
    }

    #[test]
    fn rejects_removed_fields() {
        for field in ["routes: [infusion(iv) -> central], ", "fa: |t| fa! {}, "] {
            let error = syn::parse_str::<OdeInput>(&format!("{field}{}", source("")))
                .err()
                .unwrap();
            assert!(error.to_string().contains("fields have been removed"));
        }
    }

    #[test]
    fn rejects_legacy_closure_and_missing_declarations() {
        let error = syn::parse_str::<OdeInput>(
            "diffeq: |x, p, t, dx, b, rateiv, cov| {}, out: |x, p, t, cov, y| {}",
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("requires `name`, `params`, `states`, and `outputs`"));
    }

    #[test]
    fn rhs_routes_share_inputs_by_kind_local_ordinal() {
        let input = syn::parse_str::<OdeInput>(&source(
            "dx[depot] = bolus[oral] + bolus[sc]; dx[central] = infusion[iv];",
        ))
        .unwrap();
        let bindings = ode_route_input_bindings(&input.routes);
        assert_eq!(dense_index_len(&bindings), 2);
        assert_eq!(
            bindings
                .iter()
                .map(|(name, index)| (name.name(), *index))
                .collect::<Vec<_>>(),
            vec![("oral".into(), 0), ("sc".into(), 1), ("iv".into(), 0)]
        );
    }

    #[test]
    fn supports_shared_labels_and_call_syntax() {
        let input = syn::parse_str::<OdeInput>(&source(
            "dx[central] = bolus(input_1) + infusion(input_1);",
        ))
        .unwrap();
        assert_eq!(input.routes.len(), 2);
        assert_eq!(dense_index_len(&ode_route_input_bindings(&input.routes)), 1);
    }

    #[test]
    fn reuses_infusion_slots_across_derivatives() {
        let input = syn::parse_str::<OdeInput>(&source(
            "dx[depot] = infusion[iv]; dx[central] = infusion[iv] * t;",
        ))
        .unwrap();
        assert_eq!(input.routes.len(), 1);
        assert_eq!(dense_index_len(&ode_route_input_bindings(&input.routes)), 1);
    }

    #[test]
    fn validates_inferred_destinations() {
        for equations in [
            "dx[unknown] = bolus[oral];",
            "dx[depot] = bolus[oral]; dx[central] = bolus[oral];",
            "let dose = bolus[oral]; dx[depot] = dose;",
            "dx[central] = infusion[iv + 1];",
        ] {
            assert!(
                syn::parse_str::<OdeInput>(&source(equations)).is_err(),
                "{equations}"
            );
        }
    }

    #[test]
    fn retains_named_binding_collision_checks() {
        let input =
            source("dx[central] = infusion[iv];").replace("params: [ke]", "params: [central]");
        assert!(syn::parse_str::<OdeInput>(&input)
            .err()
            .unwrap()
            .to_string()
            .contains("named parameter binding `central` conflicts"));
    }
}
