use crate::simulator::likelihood::Prediction;

#[derive(Debug, Clone)]
pub struct ErrorModel<'a> {
    c: (f64, f64, f64, f64),
    gl: f64,
    e_type: &'a ErrorType,
}

impl<'a> ErrorModel<'a> {
    pub fn new(c: (f64, f64, f64, f64), gl: f64, e_type: &'a ErrorType) -> Self {
        Self { c, gl, e_type }
    }
}

#[derive(Debug, Clone)]
pub enum ErrorType {
    Add,
    Prop,
}
#[allow(clippy::extra_unused_lifetimes)]
impl<'a> ErrorModel<'_> {
    pub(crate) fn estimate_sigma(&self, prediction: &Prediction) -> f64 {
        let (c0, c1, c2, c3) = match prediction.errorpoly() {
            Some((c0, c1, c2, c3)) => (c0, c1, c2, c3),
            None => (self.c.0, self.c.1, self.c.2, self.c.3),
        };
        let alpha = c0
            + c1 * prediction.observation()
            + c2 * prediction.observation().powi(2)
            + c3 * prediction.observation().powi(3);

        let res = match self.e_type {
            ErrorType::Add => (alpha.powi(2) + self.gl.powi(2)).sqrt(),
            ErrorType::Prop => self.gl * alpha,
        };

        if res.is_nan() || res < 0.0 {
            panic!("negative sd");
            // tracing::error!(
            //     "The computed standard deviation is either NaN or negative (SD = {}), coercing to 0",
            //     res
            // );
            // 0.0
        } else {
            res
        }
    }
}
