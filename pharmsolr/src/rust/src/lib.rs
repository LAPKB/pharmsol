//! extendr bindings for pharmsol's JIT-compiled model API.
//!
//! Two functions are exposed to R:
//!
//! - `compile_model(text)` — JIT-compile a textual model definition, returning
//!   an opaque external pointer to the compiled `JitOde`.
//! - `simulate_subject(model, params, times, evids, amts, durs, cmts, outeqs,
//!    cov_names, cov_times, cov_values)` — build a single subject from event
//!   columns and a list of covariates and run the model, returning a list with
//!   `time` and `pred` numeric vectors.
//!
//! The R-side `R/pharmsolr.R` file presents a friendlier `data.frame`-based
//! API on top of these.

use extendr_api::prelude::*;
use extendr_api::Result;
use std::sync::Arc;

use pharmsol::jit::{JitOde, Model};
use pharmsol::prelude::*;
use pharmsol::Predictions;

/// Wrapper that lives behind an extendr ExternalPtr.
struct CompiledModel {
    ode: JitOde,
}

/// Compile a text model definition into a JIT'd ODE model handle.
///
/// Returns an external-pointer SEXP. The pointer is reference-counted via
/// `Arc<CompiledModel>` so cloning the R object is cheap.
#[extendr]
fn compile_model(text: &str) -> Robj {
    or_throw(compile_model_inner(text))
}

fn compile_model_inner(text: &str) -> Result<Robj> {
    let model = Model::from_text(text).map_err(|e| Error::Other(format!("parse error: {e}")))?;
    let ode = model
        .compile()
        .map_err(|e| Error::Other(format!("compile error: {e}")))?;
    let boxed = Arc::new(CompiledModel { ode });
    Ok(ExternalPtr::new(boxed).into())
}

/// Return the compartment names of a compiled model in declaration order.
/// Index `i` (zero-based) is the value used as `cmt` in the events table for
/// a bolus or infusion targeting this compartment.
#[extendr]
fn model_compartments(model: Robj) -> Robj {
    or_throw(model_names(model, |c| c.compartments().to_vec()))
}

/// Return the output names of a compiled model in declaration order. Index
/// `i` (zero-based) is the value used as `outeq` in the events table for
/// observations of this output.
#[extendr]
fn model_outputs(model: Robj) -> Robj {
    or_throw(model_names(model, |c| c.outputs().to_vec()))
}

/// Return the parameter names of a compiled model in the order expected by
/// the `params` argument of [`simulate_subject`].
#[extendr]
fn model_params(model: Robj) -> Robj {
    or_throw(model_names(model, |c| c.params().to_vec()))
}

/// Return the covariate names referenced by a compiled model.
#[extendr]
fn model_covariates(model: Robj) -> Robj {
    or_throw(model_names(model, |c| c.covariates().to_vec()))
}

fn model_names(model: Robj, pick: impl Fn(&JitOde) -> Vec<String>) -> Result<Robj> {
    let ptr: ExternalPtr<Arc<CompiledModel>> = model
        .try_into()
        .map_err(|_| Error::Other("`model` is not a pharmsolr model handle".into()))?;
    let compiled: &Arc<CompiledModel> = &*ptr;
    let names = pick(&compiled.ode);
    Ok(Strings::from_values(names).into())
}

/// Run a single subject through a compiled model.
///
/// All event-shape arguments are parallel vectors of equal length; covariates
/// are passed as three parallel R objects: a character vector of names, a list
/// of numeric `time` vectors, and a list of numeric `value` vectors.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn simulate_subject(
    model: Robj,
    params: Doubles,
    times: Doubles,
    evids: Integers,
    amts: Doubles,
    durs: Doubles,
    cmts: Integers,
    outeqs: Integers,
    cov_names: Strings,
    cov_times: List,
    cov_values: List,
) -> Robj {
    or_throw(simulate_subject_inner(
        model, params, times, evids, amts, durs, cmts, outeqs, cov_names, cov_times, cov_values,
    ))
}

#[allow(clippy::too_many_arguments)]
fn simulate_subject_inner(
    model: Robj,
    params: Doubles,
    times: Doubles,
    evids: Integers,
    amts: Doubles,
    durs: Doubles,
    cmts: Integers,
    outeqs: Integers,
    cov_names: Strings,
    cov_times: List,
    cov_values: List,
) -> Result<Robj> {
    let ptr: ExternalPtr<Arc<CompiledModel>> = model
        .try_into()
        .map_err(|_| Error::Other("`model` is not a pharmsolr model handle".into()))?;
    let compiled: &Arc<CompiledModel> = &*ptr;

    let n_events = times.len();
    if evids.len() != n_events
        || amts.len() != n_events
        || durs.len() != n_events
        || cmts.len() != n_events
        || outeqs.len() != n_events
    {
        return Err(Error::Other(
            "event columns must all have the same length".into(),
        ));
    }
    if cov_names.len() != cov_times.len() || cov_names.len() != cov_values.len() {
        return Err(Error::Other(
            "cov_names, cov_times, cov_values must have the same length".into(),
        ));
    }

    let mut builder = Subject::builder("subject");

    // Covariates first so they're available for the whole occasion.
    for i in 0..cov_names.len() {
        let name_owned = cov_names.elt(i).as_ref().to_string();
        let ts = cov_times.elt(i)?;
        let vs = cov_values.elt(i)?;
        let ts: Doubles = ts
            .try_into()
            .map_err(|_| Error::Other(format!("cov_times[[{}]] is not numeric", i + 1)))?;
        let vs: Doubles = vs
            .try_into()
            .map_err(|_| Error::Other(format!("cov_values[[{}]] is not numeric", i + 1)))?;
        if ts.len() != vs.len() {
            return Err(Error::Other(format!(
                "covariate {:?}: time and value have different lengths",
                name_owned
            )));
        }
        for (t, v) in ts.iter().zip(vs.iter()) {
            builder = builder.covariate(&name_owned, t.0, v.0);
        }
    }

    // Events.
    for i in 0..n_events {
        let t = times[i].0;
        let amt = amts[i].0;
        let dur = durs[i].0;
        let evid = evids[i].0;
        let cmt = i32_to_usize(cmts[i].0, "cmt", i)?;
        let outeq = i32_to_usize(outeqs[i].0, "outeq", i)?;
        builder = match evid {
            // EVID=1 is a dose; pharmsol distinguishes bolus from infusion
            // by whether `dur` is zero (matches NONMEM semantics).
            1 => {
                if dur > 0.0 {
                    builder.infusion(t, amt, cmt, dur)
                } else {
                    builder.bolus(t, amt, cmt)
                }
            }
            0 => builder.observation(t, 0.0, outeq),
            other => {
                return Err(Error::Other(format!(
                    "row {}: invalid evid {} (expected 0 or 1)",
                    i + 1,
                    other
                )));
            }
        };
    }

    let subject = builder.build();
    let p: Vec<f64> = params.iter().map(|x| x.0).collect();

    let (preds, _) = compiled
        .ode
        .simulate_subject(&subject, &p, None)
        .map_err(|e| Error::Other(format!("simulation error: {e}")))?;

    let v = preds.get_predictions();
    let out_times: Doubles = v.iter().map(|p| Rfloat::from(p.time())).collect();
    let out_preds: Doubles = v.iter().map(|p| Rfloat::from(p.prediction())).collect();
    Ok(list!(time = out_times, pred = out_preds).into())
}

/// Convert an `extendr_api::Result` into either its `Ok` value or a clean
/// R-side error via `throw_r_error`. Without this, returning `Err(...)` from
/// an `#[extendr]` function panics in Rust (see extendr-api's default
/// `From<Result<_,_>> for Robj` impl), which surfaces in R as a noisy
/// `explicit panic` rather than a normal condition.
fn or_throw<T: Into<Robj>>(res: Result<T>) -> Robj {
    match res {
        Ok(v) => v.into(),
        Err(e) => throw_r_error(e.to_string()),
    }
}

#[inline]
fn i32_to_usize(v: i32, label: &str, row: usize) -> Result<usize> {
    if v < 0 {
        Err(Error::Other(format!(
            "row {}: `{}` must be non-negative, got {}",
            row + 1,
            label,
            v
        )))
    } else {
        Ok(v as usize)
    }
}

extendr_module! {
    mod pharmsolr;
    fn compile_model;
    fn simulate_subject;
    fn model_compartments;
    fn model_outputs;
    fn model_params;
    fn model_covariates;
}
