#[repr(C)]
#[derive(Debug, Clone)]
/// This structs holds the metadata of the model
pub struct Meta {
    params: Vec<String>,
}

impl Meta {
    /// Create a new Meta struct
    pub fn new(params: Vec<&str>) -> Self {
        let params = params.iter().map(|x| x.to_string()).collect();
        Meta { params }
    }

    /// Get the parameters of the model
    pub fn get_params(&self) -> &Vec<String> {
        &self.params
    }
}
