use std::collections::HashMap;

// Maximum number of drugs you expect
pub const MAX_DRUGS: usize = 10;

///Mappings maps 'INPUT' in the data into 'CMT' in the model
/// It provides a way to define the data independent on the characteristics of the model
/// Mappings are optional, if not provided, INPUT is assumed to be equal to CMT
///
/// As of this moment, Mappings are only relevant for Bolus dosing.
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

    pub fn insert(&mut self, input: usize, cmt: usize) -> Result<(), &str> {
        if input < MAX_DRUGS {
            if self.mappings[input].is_none() {
                self.count += 1;
            }
            self.mappings[input] = Some(cmt);
            Ok(())
        } else {
            Err("Input exceeds maximum number of drugs")
        }
    }

    pub fn get(&self, input: usize) -> Option<usize> {
        if input < MAX_DRUGS {
            self.mappings[input]
        } else {
            None
        }
    }

    pub fn to_hashmap(&self) -> HashMap<usize, usize> {
        let mut map = HashMap::new();
        for (input, &mapping) in self.mappings.iter().enumerate() {
            if let Some(cmt) = mapping {
                map.insert(input, cmt);
            }
        }
        map
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}
