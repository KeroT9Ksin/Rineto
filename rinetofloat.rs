//! RinetoFloat16 и RinetoFloat8: форматы с плавающей точкой
//! оптимизированные под распределение весов LLM N(0, 0.02)

use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct RinetoFloat16(u16);

#[pymethods]
impl RinetoFloat16 {
    #[staticmethod]
    fn from_f32(v: f32) -> Self {
        Self(crate::rinetofloat::RinetoFloat16::from_f32(v).0)
    }

    fn to_f32(&self) -> f32 {
        crate::rinetofloat::RinetoFloat16(self.0).to_f32()
    }
}

#[pyclass]
pub struct RinetoMatrixF16 {
    shape: (usize, usize),
    data: Vec<RinetoFloat16>,
}

#[pymethods]
impl RinetoMatrixF16 {
    #[staticmethod]
    fn from_f32(data: Vec<f32>, rows: usize, cols: usize) -> PyResult<Self> {
        if data.len() != rows * cols {
            return Err(pyo3::exceptions::PyValueError::new_err("Неверный размер"));
        }
        let quantized = data.into_iter()
            .map(|v| RinetoFloat16(crate::rinetofloat::RinetoFloat16::from_f32(v).0))
            .collect();
        Ok(Self { shape: (rows, cols), data: quantized })
    }

    fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|v| crate::rinetofloat::RinetoFloat16(v.0).to_f32()).collect()
    }

    fn matmul(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.shape.1 {
            return Err(pyo3::exceptions::PyValueError::new_err("Неверная размерность"));
        }
        let (rows, cols) = self.shape;
        let mut out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols {
                let w = crate::rinetofloat::RinetoFloat16(self.data[i * cols + j].0).to_f32();
                sum += x[j] * w;
            }
            out[i] = sum;
        }
        Ok(out)
    }
}