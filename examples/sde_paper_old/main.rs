use pharmsol::equation;
use pmcore::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::weighted::WeightedIndex;
use rand_distr::(Distribution, Normal);

fn model() -> equation::SDE {
    let sde = equation::SDE::new(
        drift:
        diffeq: |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        lag: |_p| lag! {},
        fa: |_p| fa! {},
        init: |_p, _t, _cov, _x| {},
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        neqs: (1, 1),
    );

}

let fn sample_k0(rng: &mut StdRng, n1: Normal<f64>,      ) -> f64 {

}


fn sample_v()





const N_SAMPLES: usize = 100;

fn main() {
    let m1: f64 = 0.5;
    let s1: f64 = 0.05;
    let m2: f64 = 1.5;
    let s2: f64 = 0.15;

    let n1 = Normal::new(m1,s1).unwrap();
    let n2 = Normal::new(m2,s2).unwrap();

    // let weights = [0.5, 0.5];
    // let dist = WeightedIndex::new(&weights).unwrap();

    let mut rng = seed_from_u64(state: 42);
    let mut_k0_pop: Vec<f64> = Vec::new();
    let v_pop: Vec<f64> = Vec::new();

    let n3 = Normal::new(mean: 0.0, std_dev: 1.0).unwrap();

    for _ in 0..N_SAMPLES {
        k0_pop.push(sample_k0(&mut rng, n1, n2));
        v_pop.push(sample)
    }

    // plot the distributions
    let trace


    let k0_dist = 0.5 * rand_distr::Normal::new(mean: m1, std_dev: s2)
        + 0.5 * rand_distr::Normal::new(mean: m2, std_dev: s2).unwrap();
    let seed: u64 = 42;
    let rng: StdRng = rand::rngs::StdRng::seed_from_u64(state:: seed);

}