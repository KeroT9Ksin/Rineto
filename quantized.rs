//! Квантованные форматы Ринето: br1, ir1, br4, ir4, br16.
//! Формула Ринето + AVX2 оптимизация.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ───────────────────────── br1 (1-bit binary) ─────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoBR1 {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub scale: f32,
    original_len: usize,
    bits: Vec<u32>,
}

#[pymethods]
impl RinetoBR1 {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(PyValueError::new_err("data не может быть пустым"));
        }
        let mut abs_vals: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scale = abs_vals[(n as f32 * 0.999) as usize].max(1e-8);

        let mut bits = Vec::new();
        for chunk in data.chunks(32) {
            let mut packed = 0u32;
            for (i, &val) in chunk.iter().enumerate() {
                if val >= 0.0 { packed |= 1 << i; }
            }
            bits.push(packed);
        }
        Ok(Self {
            shape,
            scale,
            original_len: data.len(),
           bits,
        })
    }

    pub fn dequantize(&self) -> PyResult<Vec<f32>> {
        let n = self.original_len;
        let mut out = vec![0.0f32; n];

        for (chunk_idx, &packed) in self.bits.iter().enumerate() {
            for bit in 0..32 {
                let idx = chunk_idx * 32 + bit;

                if idx >= n {
                    break;
                }

                let sign = if (packed >> bit) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                };

                out[idx] = sign * self.scale;
            }
        }

        Ok(out)
    }

    pub fn matmul(&self, x: Vec<f32>, in_features: usize, out_features: usize) -> PyResult<Vec<f32>> {
        if x.len() % in_features != 0 {
            return Err(PyValueError::new_err("x length must be multiple of in_features"));
        }
        let batch = x.len() / in_features;
        let mut out = vec![0.0f32; batch * out_features];
        let w_f32 = self.dequantize()?;

        for b in 0..batch {
            let x_row = &x[b * in_features..(b + 1) * in_features];
            for o in 0..out_features {
                let w_row = &w_f32[o * in_features..(o + 1) * in_features];
                let dot: f32 = x_row.iter().zip(w_row).map(|(a, b)| a * b).sum();
                out[b * out_features + o] = dot;
            }
        }
        Ok(out)
    }
}

// ───────────────────────── ir1 (1-bit integer) ─────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoIR1 {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub min_val: f32,
    #[pyo3(get)]
    pub max_val: f32,
    bits: Vec<u32>,
}

#[pymethods]
impl RinetoIR1 {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(PyValueError::new_err("data не может быть пустым"));
        }
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = (min_val + max_val) / 2.0;

        let mut bits = Vec::new();
        for chunk in data.chunks(32) {
            let mut packed = 0u32;
            for (i, &val) in chunk.iter().enumerate() {
                if val >= threshold { packed |= 1 << i; }
            }
            bits.push(packed);
        }
        Ok(Self { shape, min_val, max_val, bits })
    }

    pub fn dequantize(&self) -> PyResult<Vec<f32>> {
        let n = self.bits.len() * 32;
        let mut out = vec![0.0f32; n];
        for (i, &packed) in self.bits.iter().enumerate() {
            for bit in 0..32 {
                let val = if (packed >> bit) & 1 == 1 { self.max_val } else { self.min_val };
                out[i * 32 + bit] = val;
            }
        }
        Ok(out)
    }
}

// ───────────────────────── br4 (4-bit balanced) ─────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoBR4 {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub scale: f32,
    original_len: usize,
    packed: Vec<u8>,
}

#[pymethods]
impl RinetoBR4 {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(PyValueError::new_err("data не может быть пустым"));
        }
        let mut abs_vals: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scale = abs_vals[(n as f32 * 0.999) as usize].max(1e-8) / 7.0;

        let mut packed = Vec::new();
        for chunk in data.chunks(2) {
            let q0 = (chunk[0] / scale).round().clamp(-7.0, 7.0) as i8;
            let q1 = if chunk.len() > 1 { (chunk[1] / scale).round().clamp(-7.0, 7.0) as i8 } else { 0 };
            let u0 = (q0 + 8) as u8;
            let u1 = (q1 + 8) as u8;
            packed.push((u1 << 4) | u0);
        }
        Ok(Self {
            shape,
            scale,
            original_len: data.len(),
           packed,
        })
    }

    pub fn dequantize(&self) -> PyResult<Vec<f32>> {
        let n = self.original_len;
        let mut out = vec![0.0f32; n];

        for (chunk_idx, &byte) in self.packed.iter().enumerate() {
            let idx0 = chunk_idx * 2;

            if idx0 < n {
                let u0 = (byte & 0x0F) as i8 - 8;
                out[idx0] = u0 as f32 * self.scale;
            }

            if idx0 + 1 < n {
                let u1 = (byte >> 4) as i8 - 8;
                out[idx0 + 1] = u1 as f32 * self.scale;
            }
        }

        Ok(out)
    }

    pub fn matmul(&self, x: Vec<f32>, in_features: usize, out_features: usize) -> PyResult<Vec<f32>> {
        if x.len() % in_features != 0 {
            return Err(PyValueError::new_err("x length must be multiple of in_features"));
        }
        let batch = x.len() / in_features;
        let mut out = vec![0.0f32; batch * out_features];
        let w_f32 = self.dequantize()?;

        for b in 0..batch {
            let x_row = &x[b * in_features..(b + 1) * in_features];
            for o in 0..out_features {
                let w_row = &w_f32[o * in_features..(o + 1) * in_features];
                let dot: f32 = x_row.iter().zip(w_row).map(|(a, b)| a * b).sum();
                out[b * out_features + o] = dot;
            }
        }
        Ok(out)
    }
}

// ───────────────────────── ir4 (4-bit integer) ─────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoIR4 {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub min_val: f32,
    #[pyo3(get)]
    pub scale: f32,
    packed: Vec<u8>,
}

#[pymethods]
impl RinetoIR4 {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(PyValueError::new_err("data не может быть пустым"));
        }
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((max_val - min_val) / 15.0).max(1e-8);

        let mut packed = Vec::new();
        for chunk in data.chunks(2) {
            let q0 = ((chunk[0] - min_val) / scale).round().clamp(0.0, 15.0) as u8;
            let q1 = if chunk.len() > 1 { ((chunk[1] - min_val) / scale).round().clamp(0.0, 15.0) as u8 } else { 0 };
            packed.push((q1 << 4) | q0);
        }
        Ok(Self { shape, min_val, scale, packed })
    }

    pub fn dequantize(&self) -> PyResult<Vec<f32>> {
        let n = self.packed.len() * 2;
        let mut out = vec![0.0f32; n];
        for (i, &byte) in self.packed.iter().enumerate() {
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;
            out[i * 2] = self.min_val + q0 * self.scale;
            out[i * 2 + 1] = self.min_val + q1 * self.scale;
        }
        Ok(out)
    }

    pub fn matmul(&self, x: Vec<f32>, in_features: usize, out_features: usize) -> PyResult<Vec<f32>> {
        if x.len() % in_features != 0 {
            return Err(PyValueError::new_err("x length must be multiple of in_features"));
        }
        let batch = x.len() / in_features;
        let mut out = vec![0.0f32; batch * out_features];
        let w_f32 = self.dequantize()?;

        for b in 0..batch {
            let x_row = &x[b * in_features..(b + 1) * in_features];
            for o in 0..out_features {
                let w_row = &w_f32[o * in_features..(o + 1) * in_features];
                let dot: f32 = x_row.iter().zip(w_row).map(|(a, b)| a * b).sum();
                out[b * out_features + o] = dot;
            }
        }
        Ok(out)
    }
}

// ───────────────────────── br16 (16-bit balanced, max quality) ─────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoBR16 {
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub scale: f32,
    data: Vec<i16>,
}

#[pymethods]
impl RinetoBR16 {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let n = data.len();
        if n == 0 {
            return Err(PyValueError::new_err("data не может быть пустым"));
        }
        let mut abs_vals: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scale = abs_vals[(n as f32 * 0.999) as usize].max(1e-8) / 32767.0;

        let quantized: Vec<i16> = data.iter()
        .map(|&x| {
            let q = (x / scale).round();
            q.clamp(-32767.0, 32767.0) as i16
        })
        .collect();

        Ok(Self { shape, scale, data: quantized })
    }

    pub fn dequantize(&self) -> PyResult<Vec<f32>> {
        Ok(self.data.iter().map(|&q| q as f32 * self.scale).collect())
    }

    pub fn matmul(&self, x: Vec<f32>, in_features: usize, out_features: usize) -> PyResult<Vec<f32>> {
        if x.len() % in_features != 0 {
            return Err(PyValueError::new_err("x length must be multiple of in_features"));
        }
        let batch = x.len() / in_features;
        let mut out = vec![0.0f32; batch * out_features];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    br16_matmul_avx2(
                        &x,
                        &self.data,
                        self.scale,
                        &mut out,
                        batch,
                        in_features,
                        out_features,
                    );
                }
                return Ok(out);
            }
        }

        // Fallback: scalar
        for b in 0..batch {
            let x_row = &x[b * in_features..(b + 1) * in_features];
            for o in 0..out_features {
                let w_row = &self.data[o * in_features..(o + 1) * in_features];
                let dot: f32 = x_row.iter().zip(w_row.iter())
                .map(|(a, b)| a * (*b as f32) * self.scale)
                .sum();
                out[b * out_features + o] = dot;
            }
        }
        Ok(out)
    }
}

// ───────────────────────── AVX2 Helper (Private, not exposed to PyO3) ─────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn br16_matmul_avx2(
    x: &[f32],
    w_data: &[i16],
    scale: f32,
    out: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) {
    use std::arch::x86_64::*;

    let scale_vec = _mm256_set1_ps(scale);

    for b in 0..batch {
        let x_row = &x[b * in_features..(b + 1) * in_features];
        let out_row = &mut out[b * out_features..(b + 1) * out_features];

        for o in 0..out_features {
            let mut acc = _mm256_setzero_ps();
            let w_row = &w_data[o * in_features..(o + 1) * in_features];

            let mut i = 0;

            while i + 8 <= in_features {
                let x_vec = _mm256_loadu_ps(x_row.as_ptr().add(i));

                let w_i16 = _mm_loadu_si128(w_row.as_ptr().add(i) as *const __m128i);
                let w_i32 = _mm256_cvtepi16_epi32(w_i16);
                let w_f32 = _mm256_cvtepi32_ps(w_i32);

                let prod = _mm256_mul_ps(x_vec, w_f32);
                acc = _mm256_fmadd_ps(prod, scale_vec, acc);

                i += 8;
            }

            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let mut sum = tmp.iter().sum::<f32>();

            while i < in_features {
                sum += x_row[i] * (w_row[i] as f32) * scale;
                i += 1;
            }

            out_row[o] = sum;
        }
    }
}
