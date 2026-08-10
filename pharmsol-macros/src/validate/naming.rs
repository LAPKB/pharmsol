//! Detection of collisions between the names a macro binds and the closure
//! parameters the user writes.

use std::collections::HashSet;
use syn::{ExprClosure, Ident};

use crate::analysis::closure_param_names;
use crate::symbols::{route_input_idents, OdeRouteDecl};

fn validate_binding_conflicts(
    left_label: &str,
    left: &[Ident],
    right_label: &str,
    right: &[Ident],
    context: &str,
) -> syn::Result<()> {
    let right_names = right.iter().map(Ident::to_string).collect::<HashSet<_>>();

    for ident in left {
        let name = ident.to_string();
        if right_names.contains(&name) {
            return Err(syn::Error::new_spanned(
                ident,
                format!(
                    "named {left_label} binding `{name}` conflicts with named {right_label} binding in {context}"
                ),
            ));
        }
    }

    Ok(())
}

fn validate_closure_param_conflicts(
    closure_label: &str,
    closure: &ExprClosure,
    bindings: &[Ident],
    binding_label: &str,
) -> syn::Result<()> {
    let parameter_names = closure_param_names(closure)
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<HashSet<_>>();

    for ident in bindings {
        let name = ident.to_string();
        if parameter_names.contains(&name) {
            return Err(syn::Error::new_spanned(
                ident,
                format!(
                    "named {binding_label} binding `{name}` conflicts with `{closure_label}` closure parameter `{name}`"
                ),
            ));
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct NamedBindingSets<'a> {
    pub(crate) params: &'a [Ident],
    pub(crate) derived: &'a [Ident],
    pub(crate) covariates: &'a [Ident],
    pub(crate) states: &'a [Ident],
    pub(crate) outputs: &'a [Ident],
    pub(crate) routes: &'a [OdeRouteDecl],
}

#[derive(Clone, Copy)]
pub(crate) struct CommonBindingClosures<'a> {
    pub(crate) lag: Option<&'a ExprClosure>,
    pub(crate) fa: Option<&'a ExprClosure>,
    pub(crate) init: Option<&'a ExprClosure>,
    pub(crate) out: &'a ExprClosure,
}

#[derive(Clone, Copy)]
pub(crate) struct AnalyticalBindingClosures<'a> {
    pub(crate) derive: Option<&'a ExprClosure>,
    pub(crate) common: CommonBindingClosures<'a>,
}

#[derive(Clone, Copy)]
pub(crate) struct OdeBindingClosures<'a> {
    pub(crate) diffeq: &'a ExprClosure,
    pub(crate) common: CommonBindingClosures<'a>,
}

#[derive(Clone, Copy)]
pub(crate) struct SdeBindingClosures<'a> {
    pub(crate) drift: &'a ExprClosure,
    pub(crate) diffusion: &'a ExprClosure,
    pub(crate) common: CommonBindingClosures<'a>,
}

pub(crate) fn validate_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: OdeBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived: _,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let OdeBindingClosures {
        diffeq,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "declaration-first `ode!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`diffeq` and `out` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`out` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`out` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "declaration-first `ode!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "declaration-first `ode!` named binding generation",
    )?;

    validate_closure_param_conflicts("diffeq", diffeq, params, "parameter")?;
    validate_closure_param_conflicts("diffeq", diffeq, covariates, "covariate")?;
    validate_closure_param_conflicts("diffeq", diffeq, states, "state")?;

    if let Some(lag) = lag {
        validate_binding_conflicts(
            "covariate",
            covariates,
            "route",
            &route_inputs,
            "`lag` named binding generation",
        )?;
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_binding_conflicts(
            "covariate",
            covariates,
            "route",
            &route_inputs,
            "`fa` named binding generation",
        )?;
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

pub(crate) fn validate_analytical_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: AnalyticalBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let AnalyticalBindingClosures {
        derive,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "covariate",
        covariates,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "derived parameter",
        derived,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;
    validate_binding_conflicts(
        "output",
        outputs,
        "route",
        &route_inputs,
        "`analytical!` named binding generation",
    )?;

    if let Some(derive) = derive {
        validate_closure_param_conflicts("derive", derive, params, "parameter")?;
        validate_closure_param_conflicts("derive", derive, derived, "derived parameter")?;
        validate_closure_param_conflicts("derive", derive, covariates, "covariate")?;
    }

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, derived, "derived parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, derived, "derived parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, derived, "derived parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, derived, "derived parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}

pub(crate) fn validate_sde_named_binding_compatibility(
    bindings: NamedBindingSets<'_>,
    closures: SdeBindingClosures<'_>,
) -> syn::Result<()> {
    let NamedBindingSets {
        params,
        derived: _,
        covariates,
        states,
        outputs,
        routes,
    } = bindings;
    let SdeBindingClosures {
        drift,
        diffusion,
        common: CommonBindingClosures { lag, fa, init, out },
    } = closures;
    let route_inputs = route_input_idents(routes);

    validate_binding_conflicts(
        "parameter",
        params,
        "covariate",
        covariates,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "state",
        states,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "state",
        states,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "covariate",
        covariates,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "parameter",
        params,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "output",
        outputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "state",
        states,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;
    validate_binding_conflicts(
        "output",
        outputs,
        "route",
        &route_inputs,
        "`sde!` named binding generation",
    )?;

    validate_closure_param_conflicts("drift", drift, params, "parameter")?;
    validate_closure_param_conflicts("drift", drift, covariates, "covariate")?;
    validate_closure_param_conflicts("drift", drift, states, "state")?;
    validate_closure_param_conflicts("diffusion", diffusion, params, "parameter")?;
    validate_closure_param_conflicts("diffusion", diffusion, states, "state")?;

    if let Some(lag) = lag {
        validate_closure_param_conflicts("lag", lag, params, "parameter")?;
        validate_closure_param_conflicts("lag", lag, covariates, "covariate")?;
        validate_closure_param_conflicts("lag", lag, &route_inputs, "route")?;
    }

    if let Some(fa) = fa {
        validate_closure_param_conflicts("fa", fa, params, "parameter")?;
        validate_closure_param_conflicts("fa", fa, covariates, "covariate")?;
        validate_closure_param_conflicts("fa", fa, &route_inputs, "route")?;
    }

    if let Some(init) = init {
        validate_closure_param_conflicts("init", init, params, "parameter")?;
        validate_closure_param_conflicts("init", init, covariates, "covariate")?;
        validate_closure_param_conflicts("init", init, states, "state")?;
    }

    validate_closure_param_conflicts("out", out, params, "parameter")?;
    validate_closure_param_conflicts("out", out, covariates, "covariate")?;
    validate_closure_param_conflicts("out", out, states, "state")?;
    validate_closure_param_conflicts("out", out, outputs, "output")?;

    Ok(())
}
