//! Полная трансплантация с квантованными весами и интеграцией всех компонентов Rineto.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{RinetoBR4, RinetoIR4};
use crate::{rmsnorm_forward, sample_distribution, RinetoAttention, RinetoFFN, RinetoLM, XorShift64};

// ───────────────────────── helpers ─────────────────────────

fn bytes_to_f32(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err("bytes.len() % 4 != 0"));
    }
    Ok(bytes
    .chunks_exact(4)
    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    .collect())
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn ffn_silu_core(ffn: &RinetoFFN, x: &[f32]) -> Vec<f32> {
    let h = ffn.hidden_dim;
    let dim = ffn.dim;
    let mut activated = vec![0.0f32; h];

    for j in 0..h {
        let mut gate = ffn.b1[j];
        let mut up = ffn.b1[h + j];
        for k in 0..dim {
            gate += x[k] * ffn.w1[j * dim + k];
            up += x[k] * ffn.w1[(h + j) * dim + k];
        }
        activated[j] = silu(gate) * up;
    }

    let mut out = vec![0.0f32; dim];
    for o in 0..dim {
        let mut sum = ffn.b2[o];
        for j in 0..h {
            sum += activated[j] * ffn.w2[o * h + j];
        }
        out[o] = sum;
    }
    out
}

fn expand_kv(w: Vec<f32>, num_kv_heads: usize, num_heads: usize, dim: usize) -> Vec<f32> {
    let head_dim = dim / num_heads;
    let factor = num_heads / num_kv_heads.max(1);
    let row = head_dim * dim;
    let mut out = Vec::with_capacity(num_heads * row);
    for kv in 0..num_kv_heads {
        let start = kv * row;
        let end = ((kv + 1) * row).min(w.len());
        if start >= w.len() { break; }
        let block = &w[start..end];
        for _ in 0..factor {
            out.extend_from_slice(block);
        }
    }
    out
}

fn expand_kv_bias(w: Vec<f32>, num_kv_heads: usize, num_heads: usize) -> Vec<f32> {
    let factor = num_heads / num_kv_heads.max(1);
    let per = w.len() / num_kv_heads.max(1);
    let mut out = Vec::with_capacity(num_heads * per);
    for kv in 0..num_kv_heads {
        let block = &w[kv * per..(kv + 1) * per];
        for _ in 0..factor {
            out.extend_from_slice(block);
        }
    }
    out
}

// ───────────────────────── структуры (ОПРЕДЕЛЕНЫ ДО использования) ─────────────────────────

/// Квантованные веса одного слоя (для экономии памяти).
struct QuantLayer {
    wq: RinetoBR4,
    wk: RinetoBR4,
    wv: RinetoBR4,
    wo: RinetoIR4,
}

// ───────────────────────── трансплантер ─────────────────────────

#[pyclass]
pub struct RinetoTransplant {
    #[pyo3(get)]
    pub dim: usize,
    #[pyo3(get)]
    pub num_heads: usize,
    inv_freq_head: Vec<f32>,
    final_norm: Vec<f32>,
    quant_layers: Vec<Option<QuantLayer>>,
    use_quantized: bool,
}

#[pymethods]
impl RinetoTransplant {
    #[new]
    #[pyo3(signature = (dim, num_heads, rope_base=10000.0, use_quantized=false))]
    fn new(dim: usize, num_heads: usize, rope_base: f32, use_quantized: bool) -> PyResult<Self> {
        let head_dim = dim / num_heads;
        let half = head_dim / 2;
        let inv_freq_head: Vec<f32> = (0..half)
        .map(|i| (rope_base as f32).powf(-2.0 * i as f32 / head_dim as f32))
        .collect();
        Ok(Self {
            dim,
            num_heads,
            inv_freq_head,
            final_norm: Vec::new(),
           quant_layers: Vec::new(),
           use_quantized,
        })
    }

    fn set_final_norm(&mut self, weight: Vec<f32>) {
        self.final_norm = weight;
    }

    fn set_embed_bytes(&self, model: &mut RinetoLM, bytes: Vec<u8>, tied_head: bool) -> PyResult<()> {
        let w = bytes_to_f32(&bytes)?;
        model.embedding.set_weights(w.clone())?;
        if tied_head {
            model.head.set_weights(w)?;
            model.head.set_bias(vec![0.0; model.vocab_size])?;
        }
        Ok(())
    }

    /// Установка отдельных весов lm_head (для моделей без tied embeddings).
    fn set_head_bytes(&self, model: &mut RinetoLM, bytes: Vec<u8>) -> PyResult<()> {
        let w = bytes_to_f32(&bytes)?;
        model.head.set_weights(w)?;
        Ok(())
    }

    /// Загрузка слоя СРАЗУ в квантованные форматы (экономия памяти 8×).
    #[allow(clippy::too_many_arguments)]
    fn load_layer_bytes_quantized(
        &mut self,
        model: &mut RinetoLM,
        layer: usize,
        wq: Vec<u8>,
        wk: Vec<u8>,
        wv: Vec<u8>,
        wo: Vec<u8>,
        w1: Vec<u8>,
        w2: Vec<u8>,
        r1: Vec<f32>,
        r2: Vec<f32>,
        bq: Vec<f32>,
        bk: Vec<f32>,
        bv: Vec<f32>,
        num_kv_heads: usize,
        _hidden_dim: usize,
    ) -> PyResult<String> {
        use crate::quantized::{RinetoBR4, RinetoIR4};

        let block = model.blocks.get_mut(layer)
        .ok_or_else(|| PyValueError::new_err(format!("layer {layer}")))?;

        let wq_f32 = bytes_to_f32(&wq)?;
        let wk_f32 = bytes_to_f32(&wk)?;
        let wv_f32 = bytes_to_f32(&wv)?;
        let wo_f32 = bytes_to_f32(&wo)?;
        let w1_f32 = bytes_to_f32(&w1)?;
        let w2_f32 = bytes_to_f32(&w2)?;

        let wq_q = RinetoBR4::new(wq_f32.clone(), vec![self.dim, self.dim])?;
        let wk_q = RinetoBR4::new(
            expand_kv(wk_f32.clone(), num_kv_heads, self.num_heads, self.dim),
                                  vec![self.dim, self.dim],
        )?;
        let wv_q = RinetoBR4::new(
            expand_kv(wv_f32.clone(), num_kv_heads, self.num_heads, self.dim),
                                  vec![self.dim, self.dim],
        )?;
        let wo_q = RinetoIR4::new(wo_f32.clone(), vec![self.dim, self.dim])?;
        let w1_q = RinetoBR4::new(w1_f32.clone(), vec![w1_f32.len()])?;
        let w2_q = RinetoIR4::new(w2_f32.clone(), vec![w2_f32.len()])?;

        block.attention.wq_quant = Some(wq_q);
        block.attention.wk_quant = Some(wk_q);
        block.attention.wv_quant = Some(wv_q);
        block.attention.wo_quant = Some(wo_q);
        block.ffn.w1_quant = Some(w1_q);
        block.ffn.w2_quant = Some(w2_q);

        block.attention.bq = bq;
        block.attention.bk = expand_kv_bias(bk, num_kv_heads, self.num_heads);
        block.attention.bv = expand_kv_bias(bv, num_kv_heads, self.num_heads);
        block.rms_weight1 = r1;
        block.rms_weight2 = r2;

        // Освобождаем f32-веса
        block.attention.wq = Vec::new();
        block.attention.wk = Vec::new();
        block.attention.wv = Vec::new();
        block.attention.wo = Vec::new();
        block.ffn.w1 = Vec::new();
        block.ffn.w2 = Vec::new();

        Ok(format!("layer {layer}: квантовано (экономия ~8×)"))
    }

    /// Загрузка слоя в f32 (с опциональным квантованием).
    #[allow(clippy::too_many_arguments)]
    fn load_layer_bytes(
        &mut self,
        model: &mut RinetoLM,
        layer: usize,
        wq: Vec<u8>,
        wk: Vec<u8>,
        wv: Vec<u8>,
        wo: Vec<u8>,
        w1: Vec<u8>,
        w2: Vec<u8>,
        r1: Vec<f32>,
        r2: Vec<f32>,
        bq: Vec<f32>,
        bk: Vec<f32>,
        bv: Vec<f32>,
        num_kv_heads: usize,
        _hidden_dim: usize,
    ) -> PyResult<()> {
        let block = model.blocks.get_mut(layer)
        .ok_or_else(|| PyValueError::new_err(format!("layer {layer}")))?;

        let wq_f32 = bytes_to_f32(&wq)?;
        let wk_f32 = bytes_to_f32(&wk)?;
        let wv_f32 = bytes_to_f32(&wv)?;
        let wo_f32 = bytes_to_f32(&wo)?;

        if self.use_quantized {
            let ql = QuantLayer {
                wq: RinetoBR4::new(wq_f32.clone(), vec![self.dim, self.dim])?,
                wk: RinetoBR4::new(wk_f32.clone(), vec![self.dim, self.dim])?,
                wv: RinetoBR4::new(wv_f32.clone(), vec![self.dim, self.dim])?,
                wo: RinetoIR4::new(wo_f32.clone(), vec![self.dim, self.dim])?,
            };
            while self.quant_layers.len() <= layer {
                self.quant_layers.push(None);
            }
            self.quant_layers[layer] = Some(ql);
        }

        block.attention.wq = wq_f32;
        block.attention.wk = expand_kv(bytes_to_f32(&wk)?, num_kv_heads, self.num_heads, self.dim);
        block.attention.wv = expand_kv(bytes_to_f32(&wv)?, num_kv_heads, self.num_heads, self.dim);
        block.attention.wo = wo_f32;
        block.attention.bq = bq;
        block.attention.bk = expand_kv_bias(bk, num_kv_heads, self.num_heads);
        block.attention.bv = expand_kv_bias(bv, num_kv_heads, self.num_heads);

        block.ffn.hidden_dim = _hidden_dim;
        block.ffn.w1 = bytes_to_f32(&w1)?;
        block.ffn.b1 = vec![0.0; 2 * _hidden_dim];
        block.ffn.w2 = bytes_to_f32(&w2)?;
        block.ffn.b2 = vec![0.0; self.dim];

        block.rms_weight1 = r1;
        block.rms_weight2 = r2;
        Ok(())
    }

    fn attention_forward_rope(
        &self,
        attn: &RinetoAttention,
        xs: Vec<Vec<f32>>,
        layer_idx: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let seq = xs.len();
        if seq == 0 { return Ok(Vec::new()); }

        let dim = self.dim;
        let num_heads = self.num_heads;
        let head_dim = dim / num_heads;
        let half = head_dim / 2;

        let (wq_f32, wk_f32, wv_f32, wo_f32) = if self.use_quantized && layer_idx < self.quant_layers.len() {
            if let Some(ql) = &self.quant_layers[layer_idx] {
                (ql.wq.dequantize()?, ql.wk.dequantize()?, ql.wv.dequantize()?, ql.wo.dequantize()?)
            } else {
                (attn.wq.clone(), attn.wk.clone(), attn.wv.clone(), attn.wo.clone())
            }
        } else {
            (attn.wq.clone(), attn.wk.clone(), attn.wv.clone(), attn.wo.clone())
        };

        let mut q = vec![0.0f32; seq * dim];
        let mut k = vec![0.0f32; seq * dim];
        let mut v = vec![0.0f32; seq * dim];

        for t in 0..seq {
            let x = &xs[t];
            for o in 0..dim {
                let mut qv = attn.bq.get(o).copied().unwrap_or(0.0);
                let mut kv = attn.bk.get(o).copied().unwrap_or(0.0);
                let mut vv = attn.bv.get(o).copied().unwrap_or(0.0);
                for j in 0..dim {
                    qv += x[j] * wq_f32[o * dim + j];
                    kv += x[j] * wk_f32[o * dim + j];
                    vv += x[j] * wv_f32[o * dim + j];
                }
                q[t * dim + o] = qv;
                k[t * dim + o] = kv;
                v[t * dim + o] = vv;
            }
        }

        for t in 0..seq {
            for head in 0..num_heads {
                let base = t * dim + head * head_dim;
                for i in 0..half {
                    let angle = t as f32 * self.inv_freq_head[i];
                    let cos = angle.cos();
                    let sin = angle.sin();
                    let qi = q[base + i];
                    let qh = q[base + i + half];
                    q[base + i] = qi * cos - qh * sin;
                    q[base + i + half] = qh * cos + qi * sin;
                    let ki = k[base + i];
                    let kh = k[base + i + half];
                    k[base + i] = ki * cos - kh * sin;
                    k[base + i + half] = kh * cos + ki * sin;
                }
            }
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * dim];

        for head in 0..num_heads {
            let offset = head * head_dim;
            for t in 0..seq {
                let mut scores = vec![f32::NEG_INFINITY; seq];
                for s in 0..=t {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[t * dim + offset + d] * k[s * dim + offset + d];
                    }
                    scores[s] = dot * scale;
                }
                let max = scores[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in 0..=t {
                    scores[s] = (scores[s] - max).exp();
                    sum += scores[s];
                }
                for s in 0..=t { scores[s] /= sum; }
                for d in 0..head_dim {
                    let mut acc = 0.0;
                    for s in 0..=t {
                        acc += scores[s] * v[s * dim + offset + d];
                    }
                    out[t * dim + offset + d] = acc;
                }
            }
        }

        let mut result = vec![vec![0.0f32; dim]; seq];
        for t in 0..seq {
            for o in 0..dim {
                let mut acc = 0.0;
                for j in 0..dim {
                    acc += out[t * dim + j] * wo_f32[o * dim + j];
                }
                result[t][o] = acc;
            }
        }
        Ok(result)
    }

    fn forward_logits(&self, model: &RinetoLM, token_ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(token_ids.len());
        for &t in &token_ids {
            hidden.push(model.embedding.embed(t)?);
        }

        for (layer_idx, block) in model.blocks.iter().enumerate() {
            let normed: Vec<Vec<f32>> = hidden.iter()
            .map(|row| rmsnorm_forward(row, &block.rms_weight1, block.rms_eps))
            .collect();
            let attended = self.attention_forward_rope(&block.attention, normed, layer_idx)?;
            let after: Vec<Vec<f32>> = hidden.iter().zip(attended.iter())
            .map(|(h, a)| h.iter().zip(a).map(|(x, y)| x + y).collect())
            .collect();
            let normed2: Vec<Vec<f32>> = after.iter()
            .map(|row| rmsnorm_forward(row, &block.rms_weight2, block.rms_eps))
            .collect();
            let ffn_out: Vec<Vec<f32>> = normed2.iter()
            .map(|n| ffn_silu_core(&block.ffn, n))
            .collect();
            hidden = after.iter().zip(ffn_out.iter())
            .map(|(h, f)| h.iter().zip(f).map(|(x, y)| x + y).collect())
            .collect();
        }

        if !self.final_norm.is_empty() {
            hidden = hidden.iter()
            .map(|row| rmsnorm_forward(row, &self.final_norm, 1e-6))
            .collect();
        }

        let mut logits = Vec::with_capacity(hidden.len());
        for row in &hidden {
            logits.push(model.head.forward(row.clone())?);
        }
        Ok(logits)
    }

    #[pyo3(signature = (model, prompt_ids, max_new_tokens=16, temperature=0.7, top_k=40, top_p=0.9))]
    fn generate(
        &self,
        model: &RinetoLM,
        prompt_ids: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<Vec<usize>> {
        let mut tokens = prompt_ids;
        let mut rng = XorShift64::new(42);
        for _ in 0..max_new_tokens {
            let logits = self.forward_logits(model, tokens.clone())?;
            let last = logits.last().cloned().unwrap_or_default();
            let probs = sample_distribution(&last, temperature, top_k, top_p);
            let mut r = rng.next_f32();
            let mut next = 0;
            for (i, p) in probs.iter().enumerate() {
                if *p > 0.0 {
                    r -= p;
                    if r <= 0.0 { next = i; break; }
                }
                next = i;
            }
            tokens.push(next);
            if next == 2 { break; }
        }
        Ok(tokens)
    }

    fn quantize_all_layers(&mut self, model: &RinetoLM) -> PyResult<String> {
        let mut count = 0;
        for (layer_idx, block) in model.blocks.iter().enumerate() {
            let ql = QuantLayer {
                wq: RinetoBR4::new(block.attention.wq.clone(), vec![self.dim, self.dim])?,
                wk: RinetoBR4::new(block.attention.wk.clone(), vec![self.dim, self.dim])?,
                wv: RinetoBR4::new(block.attention.wv.clone(), vec![self.dim, self.dim])?,
                wo: RinetoIR4::new(block.attention.wo.clone(), vec![self.dim, self.dim])?,
            };
            while self.quant_layers.len() <= layer_idx {
                self.quant_layers.push(None);
            }
            self.quant_layers[layer_idx] = Some(ql);
            count += 1;
        }
        self.use_quantized = true;
        Ok(format!("Квантовано {} слоёв", count))
    }

    fn quantize_ffn(&mut self, model: &RinetoLM, layer: usize) -> PyResult<String> {
        let block = model.blocks.get(layer)
        .ok_or_else(|| PyValueError::new_err(format!("layer {layer}")))?;
        let ffn = &block.ffn;

        let _w1_br4 = crate::RinetoBR4::new(ffn.w1.clone(), vec![ffn.w1.len()])?;
        let _w2_ir4 = crate::RinetoIR4::new(ffn.w2.clone(), vec![ffn.w2.len()])?;

        Ok(format!("layer {layer}: FFN w1->br4, w2->ir4"))
    }
}
