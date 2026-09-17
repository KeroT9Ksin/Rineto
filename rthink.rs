use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub const RTHINK_POINTS: [&str; 6] =
    ["цель", "план", "мысль", "сомнение", "проверка", "рефлексия"];
pub const RTHINK_KINDS: [&str; 10] = [
    "логический", "технический", "креативный", "критический", "эмоциональный",
    "пространственный", "аналоговый", "стратегический", "рефлексивный", "исследовательский",
];

#[pyclass]
#[derive(Clone, Debug)]
pub struct RThinkPoint {
    #[pyo3(get)] pub name: String,
    #[pyo3(get, set)] pub content: String,
    #[pyo3(get, set)] pub confidence: f32,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RThinkSession {
    #[pyo3(get)] pub theme: String,
    #[pyo3(get)] pub kind: String,
    #[pyo3(get)] pub points: Vec<RThinkPoint>,
    #[pyo3(get, set)] pub conspectus: String,
}
#[pymethods]
impl RThinkSession {
    #[new]
    fn new(theme: String, kind: String) -> Self {
        Self {
            theme, kind,
            points: RTHINK_POINTS.iter()
                .map(|n| RThinkPoint { name: n.to_string(), content: String::new(), confidence: 0.0 })
                .collect(),
            conspectus: String::new(),
        }
    }
    fn set_point(&mut self, idx: usize, content: String, confidence: f32) -> PyResult<()> {
        if idx >= 6 { return Err(PyValueError::new_err("RThink имеет ровно 6 точек")); }
        self.points[idx].content = content;
        self.points[idx].confidence = confidence.clamp(0.0, 1.0);
        Ok(())
    }
    /// Собирает конспект из заполненных точек.
    fn consolidate(&mut self) -> String {
        let mut s = format!("[{}] {}: ", self.kind, self.theme);
        for p in &self.points {
            if !p.content.is_empty() { s += &format!("{}={}; ", p.name, p.content); }
        }
        self.conspectus = s.clone();
        s
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RThink {
    sessions: Vec<RThinkSession>,
    #[pyo3(get)] pub total: u64,
}
#[pymethods]
impl RThink {
    #[new]
    fn new() -> Self { Self::default() }
    fn think(&mut self, theme: String, kind: String) -> usize {
        self.sessions.push(RThinkSession::new(theme, kind));
        self.total += 1;
        self.sessions.len() - 1
    }
    /// Температура сэмплирования по виду мышления.
    fn temperature_for(&self, kind: String) -> f32 {
        match kind.as_str() {
            "логический" => 0.3, "технический" => 0.4, "критический" => 0.5,
            "креативный" => 1.2, "исследовательский" => 1.0, "эмоциональный" => 0.9,
            _ => 0.8,
        }
    }
    fn session(&self, idx: usize) -> PyResult<RThinkSession> {
        self.sessions.get(idx).cloned()
            .ok_or_else(|| PyValueError::new_err("Сессия не найдена"))
    }
    fn conspectus_of(&self, idx: usize) -> PyResult<String> {
        Ok(self.sessions.get(idx).map(|s| s.conspectus.clone()).unwrap_or_default())
    }
}