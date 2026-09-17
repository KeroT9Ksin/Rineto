use pyo3::prelude::*;
use crate::{RinetoLM, RinetoRouter, DOMAIN_CODE, DOMAIN_MATH, DOMAIN_LOGIC};

#[pyclass]
pub struct RinetoCascade {
    router: RinetoRouter,
    chat: RinetoLM,
    think: RinetoLM,
    #[pyo3(get)] pub escalations: u64,
}

#[pymethods]
impl RinetoCascade {
    #[new]
    #[pyo3(signature = (vocab_size, dim=64, layers=2, heads=2))]
    fn new(vocab_size: usize, dim: usize, layers: usize, heads: usize) -> PyResult<Self> {
        let mk = |seed: u64| -> PyResult<RinetoLM> {
            RinetoLM::new(
                vocab_size, dim, layers, heads,
                0.01, 0.02, seed,
                false, 0, true, "swiglu", false, 0, false,
            )
        };
        Ok(Self {
            router: RinetoRouter::new(),
           chat: mk(1)?,
           think: mk(2)?,
           escalations: 0,
        })
    }

    /// Выбирает модель по домену: код/матем/логика -> think, иначе -> chat.
    /// Возвращает String (НЕ PyResult), поэтому `?` не нужен.
    fn select(&mut self, text: String) -> String {
        let d = self.router.route(text);
        match d.domain.as_str() {
            DOMAIN_CODE | DOMAIN_MATH | DOMAIN_LOGIC => "think".to_string(),
            _ => "chat".to_string(),
        }
    }

    fn generate(&mut self, text: String, ids: Vec<usize>, max_new: usize) -> PyResult<Vec<usize>> {
        // ИСПРАВЛЕНО: select возвращает String, убираем `?`
        let which = self.select(text);
        if which == "think" {
            self.escalations += 1;
            self.think.generate(ids, max_new, 0.7, 0, 0.9, 1.0)
        } else {
            self.chat.generate(ids, max_new, 0.8, 0, 0.9, 1.0)
        }
    }
}
