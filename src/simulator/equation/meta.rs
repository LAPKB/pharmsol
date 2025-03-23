#[repr(C)]
#[derive(Debug, Clone)]
/// Model metadata container.
///
/// This structure holds the metadata associated with a pharmacometric model,
/// including parameter names and other model-specific information that needs
/// to be preserved across simulation and estimation activities.
///
/// # Examples
///
/// ```
/// use pharmsol::simulator::equation::Meta;
///
/// let model_metadata = Meta::new(vec!["CL", "V", "KA"]);
/// assert_eq!(model_metadata.get_params().len(), 3);
/// ```
pub struct Meta {
    params: Vec<String>,
}

impl Meta {
    /// Creates a new metadata container with the specified parameter names.
    ///
    /// # Arguments
    ///
    /// * `params` - A vector of string slices representing parameter names
    ///
    /// # Returns
    ///
    /// A new `Meta` instance containing the converted parameter names
    ///
    /// # Examples
    ///
    /// ```
    /// use pharmsol::simulator::equation::Meta;
    ///
    /// let metadata = Meta::new(vec!["CL", "V", "KA"]);
    /// ```
    pub fn new(params: Vec<&str>) -> Self {
        let params = params.iter().map(|x| x.to_string()).collect();
        Meta { params }
    }

    /// Retrieves the parameter names stored in this metadata container.
    ///
    /// # Returns
    ///
    /// A reference to the vector of parameter names
    ///
    /// # Examples
    ///
    /// ```
    /// use pharmsol::simulator::equation::Meta;
    ///
    /// let metadata = Meta::new(vec!["CL", "V", "KA"]);
    /// let params = metadata.get_params();
    /// assert_eq!(params[0], "CL");
    /// assert_eq!(params[1], "V");
    /// assert_eq!(params[2], "KA");
    /// ```
    pub fn get_params(&self) -> &Vec<String> {
        &self.params
    }
}
