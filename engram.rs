use pyo3::prelude::*;
use std::collections::HashMap;
use crate::RinetoHash;

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoEngram {
    table: HashMap<u64, Vec<f32>>,
    #[pyo3(get)] pub size: usize,
}
#[pymethods]
impl RinetoEngram {
    #[new]
    fn new() -> Self { Self::default() }
    fn store(&mut self, text: String, vector: Vec<f32>) {
        self.table.insert(RinetoHash::rineto_hash(text), vector);
        self.size = self.table.len();
    }
    fn lookup(&self, text: String) -> Option<Vec<f32>> {
        self.table.get(&RinetoHash::rineto_hash(text)).cloned()
    }
}