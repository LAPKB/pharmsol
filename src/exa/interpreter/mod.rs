use std::fs;
use std::io;
use std::path::PathBuf;

use serde::Deserialize;

use crate::simulator::equation::{Meta, ODE};

#[derive(Deserialize, Debug)]
struct IrFile {
    ir_version: Option<String>,
    kind: Option<String>,
    params: Option<Vec<String>>,
    model_text: Option<String>,
}

/// Loads a very small prototype IR-based ODE and returns an `ODE` and `Meta`.
///
/// This is a pragmatic prototype implementation intended to unblock the
/// examples and tests. It only supports a single-state, single-output
/// one-compartment model where parameter 0 = ke and parameter 1 = v, and
/// where the derivative is dx0 = -ke * x0 + rateiv[0], and output is x0 / v.
///
/// The goal is to provide a working interpreter hook; a full interpreter
/// that parses `model_text` and evaluates arbitrary equations should replace
/// this prototype in the next iteration.
pub fn load_ir_ode(ir_path: PathBuf) -> Result<(ODE, Meta), io::Error> {
    let contents = fs::read_to_string(&ir_path)?;
    let ir: IrFile = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("serde_json: {}", e)))?;

    // Build Meta from params if present
    let params = match ir.params {
        Some(p) => p,
        None => Vec::new(),
    };

    // Create a simple metadata container expected by the rest of the code
    let meta = Meta::new(params.iter().map(|s| s.as_str()).collect());

    // Prototype closures for the simplest one-compartment ODE
    use crate::simulator::{T, V};
    use crate::data::Covariates;
    use diffsol::Vector; // bring trait into scope for .len()

    // DiffEq: fn(&V, &V, T, &mut V, V, V, &Covariates)
    fn diffeq(x: &V, p: &V, _t: T, dx: &mut V, _bolus: V, rateiv: V, _cov: &Covariates) {
        // Expect p[0] = ke
        let ke = if p.len() > 0 { p[0] } else { 0.0 };
        dx[0] = -ke * x[0] + rateiv[0];
    }

    // Lag: fn(&V, T, &Covariates) -> HashMap<usize, T>
    fn lag(_p: &V, _t: T, _cov: &Covariates) -> std::collections::HashMap<usize, T> {
        std::collections::HashMap::new()
    }

    // Fa: fn(&V, T, &Covariates) -> HashMap<usize, T>
    fn fa(_p: &V, _t: T, _cov: &Covariates) -> std::collections::HashMap<usize, T> {
        std::collections::HashMap::new()
    }

    // Init: fn(&V, T, &Covariates, &mut V)
    fn init(_p: &V, _t: T, _cov: &Covariates, _x: &mut V) {
        // Leave initial state as zero by default
    }

    // Out: fn(&V, &V, T, &Covariates, &mut V)
    fn out(x: &V, p: &V, _t: T, _cov: &Covariates, y: &mut V) {
        let v = if p.len() > 1 { p[1] } else { 1.0 };
        y[0] = x[0] / v;
    }

    // Construct ODE with 1 state and 1 output
    let ode = ODE::new(diffeq, lag, fa, init, out, (1_usize, 1_usize));

    Ok((ode, meta))
}
