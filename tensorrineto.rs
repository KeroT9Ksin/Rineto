//! TensorRineto («ТензорРинето»): тензор, слитый с математикой Ринето.
//!
//! Связи с ядром lib.rs:
//! - dot_simd (AVX2+FMA)        → matmul;
//! - vector_signature           → подпись для кэша и поиска;
//! - normalize_to_sphere        → сферическая нормализация;
//! - RinetoMMapWeights          → u16-квантование/деквантование;
//! - сверхпроводящий кэш        → повторный matmul за O(1).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{dot_simd, normalize_to_sphere, vector_signature, RinetoMMapWeights};

const TR_CACHE_SIZE: usize = 256;

#[pyclass]
#[derive(Clone, Debug)]
pub struct TensorRineto {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub signature: u64,
    #[pyo3(get)]
    pub is_quantized: bool,
    #[pyo3(get)]
    pub cache_hits: u64,
    #[pyo3(get)]
    pub cache_misses: u64,
    data: Vec<f32>,
    q_data: Vec<u16>,
    q_scale: f32,
    cache: Vec<Option<(u64, Vec<usize>, Vec<f32>)>>,
}

#[pymethods]
impl TensorRineto {
    #[new]
    #[pyo3(signature = (data, shape))]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let numel: usize = shape.iter().product::<usize>().max(1);
        if data.len() != numel {
            return Err(PyValueError::new_err(format!(
                "data ({} эл.) не совпадает с произведением shape ({numel})",
                data.len()
            )));
        }
        Ok(Self {
            signature: vector_signature(&data),
            shape,
            is_quantized: false,
            cache_hits: 0,
            cache_misses: 0,
            data,
            q_data: Vec::new(),
            q_scale: 0.0,
            cache: vec![None; TR_CACHE_SIZE],
        })
    }

    /// Быстрая загрузка из little-endian f32 bytes (мост к трансплантации).
    #[staticmethod]
    fn from_bytes(bytes: Vec<u8>, shape: Vec<usize>) -> PyResult<Self> {
        if bytes.len() % 4 != 0 {
            return Err(PyValueError::new_err("bytes.len() % 4 != 0"));
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::new(data, shape)
    }

    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let d = self.data()?;
        let mut out = Vec::with_capacity(d.len() * 4);
        for v in d {
            out.extend_from_slice(&v.to_le_bytes());
        }
        Ok(out)
    }

    fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Данные списком (с деквантованием, если квантован).
    fn data(&self) -> PyResult<Vec<f32>> {
        if self.is_quantized {
            RinetoMMapWeights::dequantize_u16(self.q_data.clone(), self.q_scale)
        } else {
            Ok(self.data.clone())
        }
    }

    /// u16-квантование (экономия памяти 2x) через математику Ринето.
    fn quantize_u16(&mut self) -> PyResult<()> {
        if self.is_quantized {
            return Ok(());
        }
        let (q, scale) = RinetoMMapWeights::quantize_u16(self.data.clone())?;
        self.q_data = q;
        self.q_scale = scale;
        self.data = Vec::new();
        self.is_quantized = true;
        Ok(())
    }

    fn dequantize(&mut self) -> PyResult<()> {
        if !self.is_quantized {
            return Ok(());
        }
        self.data = RinetoMMapWeights::dequantize_u16(self.q_data.clone(), self.q_scale)?;
        self.q_data = Vec::new();
        self.is_quantized = false;
        Ok(())
    }

    /// Нормализация на единичную сферу (сферическая математика Ринето).
    fn sphere(&mut self) -> PyResult<()> {
        let d = self.data()?;
        self.data = normalize_to_sphere(&d)?;
        self.signature = vector_signature(&self.data);
        Ok(())
    }

    /// Транспонирование 2D.
    fn transpose(&self) -> PyResult<TensorRineto> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err("transpose требует 2D"));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let a = self.data()?;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a[i * n + j];
            }
        }
        TensorRineto::new(out, vec![n, m])
    }

    /// Matmul [M,K] x [K,N] на AVX2 + сверхпроводящий кэш результата.
    fn matmul(&mut self, other: &TensorRineto) -> PyResult<TensorRineto> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(PyValueError::new_err("matmul требует 2D тензоры"));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(PyValueError::new_err(format!("K не совпадает: {k} и {k2}")));
        }
        let a = self.data()?;
        let b = other.data()?;

        // Ключ кэша: подписи обоих тензоров + размерность.
        let key = self
            .signature
            .wrapping_mul(0x9E3779B97F4A7C15)
            ^ other.signature
            ^ (n as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
        let slot = (key as usize) % TR_CACHE_SIZE;
        if let Some((sig, shape, res)) = &self.cache[slot] {
            if *sig == key && *shape == vec![m, n] {
                self.cache_hits += 1;
                return TensorRineto::new(res.clone(), shape.clone());
            }
        }
        self.cache_misses += 1;

        // Транспонируем B для непрерывных dot_simd по строкам.
        let mut bt = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let mut out = vec![0.0f32; m * n];
        if m >= 8 {
            use rayon::prelude::*;
            out.par_chunks_mut(n).enumerate().for_each(|(row, r)| {
                let a_row = &a[row * k..(row + 1) * k];
                for j in 0..n {
                    r[j] = dot_simd(a_row, &bt[j * k..(j + 1) * k]);
                }
            });
        } else {
            for row in 0..m {
                let a_row = &a[row * k..(row + 1) * k];
                for j in 0..n {
                    out[row * n + j] = dot_simd(a_row, &bt[j * k..(j + 1) * k]);
                }
            }
        }
        self.cache[slot] = Some((key, vec![m, n], out.clone()));
        TensorRineto::new(out, vec![m, n])
    }

    /// Поэлементное сложение.
    fn add(&self, other: &TensorRineto) -> PyResult<TensorRineto> {
        let a = self.data()?;
        let b = other.data()?;
        if a.len() != b.len() {
            return Err(PyValueError::new_err("Разная длина для add"));
        }
        TensorRineto::new(
            a.iter().zip(b).map(|(x, y)| x + y).collect(),
            self.shape.clone(),
        )
    }

    /// silu-активация (как в Qwen/Nemotron).
    fn silu(&self) -> PyResult<TensorRineto> {
        let a = self.data()?;
        TensorRineto::new(
            a.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
            self.shape.clone(),
        )
    }

    /// RMSNorm по последней оси (математика Llama/Rineto).
    #[pyo3(signature = (weight, epsilon=1e-6))]
    fn rmsnorm(&self, weight: Vec<f32>, epsilon: f32) -> PyResult<TensorRineto> {
        let dim = *self.shape.last().unwrap_or(&0);
        if weight.len() != dim || dim == 0 {
            return Err(PyValueError::new_err("weight не совпадает с последней осью"));
        }
        let a = self.data()?;
        let mut out = Vec::with_capacity(a.len());
        for row in a.chunks_exact(dim) {
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rsqrt = 1.0 / (mean_sq + epsilon).sqrt();
            for (i, v) in row.iter().enumerate() {
                out.push(v * rsqrt * weight[i]);
            }
        }
        TensorRineto::new(out, self.shape.clone())
    }

    fn get_cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorRineto(shape={:?}, sig={:#018x}, quantized={})",
            self.shape, self.signature, self.is_quantized
        )
    }
}