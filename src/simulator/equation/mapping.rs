use thiserror::Error;

// Maximum number of drugs you expect
pub(crate) const MAX_DRUGS: usize = 10;

/// Mappings maps 'INPUT' in the data into 'CMT' in the model
/// It provides a way to define the data independent on the characteristics of the model
/// Mappings are optional, if not provided, INPUT is assumed to be equal to CMT
///
/// As of this moment, [Mappings] are only relevant for Bolus dosing.
/// Infusions use the RATEIV variable in the model so no mapping is needed.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Mappings {
    mappings: [Option<usize>; MAX_DRUGS],
    // Track how many mappings are actually used
    count: usize,
}

impl Mappings {
    pub fn new() -> Self {
        Self {
            mappings: [None; MAX_DRUGS],
            count: 0,
        }
    }

    pub(crate) fn insert(&mut self, input: usize, cmt: usize) -> Result<(), MappingsError> {
        if input < MAX_DRUGS {
            if self.mappings[input].is_none() {
                self.count += 1;
            }
            self.mappings[input] = Some(cmt);
            Ok(())
        } else {
            Err(MappingsError::InputExceedsMaxDrugs(input))
        }
    }

    pub(crate) fn get(&self, input: usize) -> Option<usize> {
        if input < MAX_DRUGS {
            self.mappings[input]
        } else {
            None
        }
    }
}

#[derive(Error, Debug, Clone)]
pub enum MappingsError {
    #[error("The input number ({0}) is above the allowed maximum of {MAX_DRUGS}")]
    InputExceedsMaxDrugs(usize),
}
