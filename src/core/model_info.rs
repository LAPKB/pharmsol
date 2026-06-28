use pharmsol_dsl::{NUMERIC_OUTPUT_PREFIX, NUMERIC_ROUTE_PREFIX};

use crate::core::metadata::RouteKind;
use crate::data::{Covariates, InputLabel, OutputLabel};
use crate::simulator::{Fa, Lag};
use crate::{Event, Occasion, PharmsolError, ValidatedModelMetadata};

/// Structural information about a model.
///
/// Provides access to dimensions, metadata, lag/bioavailability closures,
/// and label resolution. Most methods have sensible defaults driven by
/// [`Self::metadata`].
pub trait ModelInfo {
    /// Number of state variables (compartments).
    fn nstates(&self) -> usize;

    /// Number of drug input routes (`bolus[]` / `rateiv[]` width).
    fn ndrugs(&self) -> usize;

    /// Number of output equations.
    fn nout(&self) -> usize;

    /// Model metadata, if attached. Drives label-name resolution.
    fn metadata(&self) -> Option<&ValidatedModelMetadata>;

    /// Lag-time closure for this model.
    fn lag(&self) -> &Lag;

    /// Fraction-absorbed (bioavailability) closure for this model.
    fn fa(&self) -> &Fa;

    // ── Provided methods ──

    /// Resolve a public input label to a dense input index.
    ///
    /// Uses metadata when available; falls back to interpreting the label as
    /// a numeric index.
    fn resolve_input(
        &self,
        label: &InputLabel,
        expected_kind: RouteKind,
    ) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            let route = metadata
                .route(label.as_str())
                .or_else(|| {
                    canonical_numeric_alias(label.as_str(), NUMERIC_ROUTE_PREFIX)
                        .and_then(|alias| metadata.route(alias.as_str()))
                })
                .ok_or_else(|| PharmsolError::UnknownInputLabel {
                    label: label.to_string(),
                })?;

            if route.kind() != expected_kind {
                return Err(PharmsolError::UnsupportedInputRouteKind {
                    input: route.input_index(),
                    kind: match expected_kind {
                        RouteKind::Bolus => pharmsol_dsl::RouteKind::Bolus,
                        RouteKind::Infusion => pharmsol_dsl::RouteKind::Infusion,
                    },
                });
            }

            return Ok(route.input_index());
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::UnknownInputLabel {
                label: label.to_string(),
            })
    }

    /// Resolve a public output label to a dense output index.
    fn resolve_output(&self, label: &OutputLabel) -> Result<usize, PharmsolError> {
        if let Some(metadata) = self.metadata() {
            return metadata
                .output_index(label.as_str())
                .or_else(|| {
                    canonical_numeric_alias(label.as_str(), NUMERIC_OUTPUT_PREFIX)
                        .and_then(|alias| metadata.output_index(alias.as_str()))
                })
                .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                    label: label.to_string(),
                });
        }

        label
            .index()
            .ok_or_else(|| PharmsolError::UnknownOutputLabel {
                label: label.to_string(),
            })
    }

    /// Resolve all events in an occasion, applying lag time and
    /// bioavailability adjustments, and mapping labels to indices.
    fn resolve_events(
        &self,
        occasion: &Occasion,
        params: &[f64],
        covariates: &Covariates,
    ) -> Result<Vec<Event>, PharmsolError> {
        let mut resolved = occasion.clone();

        for event in resolved.events_iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    let input = self.resolve_input(bolus.input(), RouteKind::Bolus)?;
                    bolus.set_input(input);
                }
                Event::Infusion(infusion) => {
                    let input = self.resolve_input(infusion.input(), RouteKind::Infusion)?;
                    infusion.set_input(input);
                }
                Event::Observation(observation) => {
                    let outeq = self.resolve_output(observation.outeq())?;
                    observation.set_outeq(outeq);
                }
            }
        }

        Ok(resolved.process_events(Some((self.fa(), self.lag(), params, covariates)), true))
    }
}

/// Build a canonical alias like `input_7` from a raw numeric label `"7"`.
fn canonical_numeric_alias(label: &str, prefix: &str) -> Option<String> {
    if label.is_empty() || !label.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(format!("{prefix}{label}"))
}
