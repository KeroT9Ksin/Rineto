use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

mod ext;

pub use ext::chat::RyzaChatEngine;
pub use ext::recorder::RinetoRecorder;
pub use ext::tensor::RinetoTensor;
pub use ext::art_color::{
    temp_to_rgb,
    rgb_to_temp,
    rgb_pixel_to_temps,
    rgb_patch_to_temp,
    temp_patch_to_rgb,
    interpolate_temp,
    image_temp_vision,
};
pub use ext::art_autoencoder::RinetoAutoencoder;
pub use ext::art_objects::segment_classes;
pub use ext::art_render2::render_scene_v2;
pub mod transplant;
pub mod tensorrineto;
pub mod quantized;
pub use quantized::{RinetoBR1, RinetoIR1, RinetoBR4, RinetoIR4, RinetoBR16};
pub mod rthink;
pub mod sferom;
pub mod engram;
pub mod cascade;

pub const DOMAIN_UNKNOWN: &str = "UNKNOWN";
pub const DOMAIN_CODE: &str = "CODE";
pub const DOMAIN_MATH: &str = "MATH";
pub const DOMAIN_LOGIC: &str = "LOGIC";
pub const DOMAIN_CHAT: &str = "CHAT";
pub const DOMAIN_PHYSICS: &str = "PHYSICS";
pub const DOMAIN_GENERAL: &str = "GENERAL";

pub const TYPE_UNKNOWN: &str = "UNKNOWN";
pub const TYPE_QUESTION: &str = "QUESTION";
pub const TYPE_FACT: &str = "FACT";
pub const TYPE_RULE: &str = "RULE";
pub const TYPE_ACTION: &str = "ACTION";
pub const TYPE_RESPONSE: &str = "RESPONSE";
pub const TYPE_ERROR: &str = "ERROR";
pub const TYPE_FORMULA: &str = "FORMULA";
pub const TYPE_CONSTRAINT: &str = "CONSTRAINT";
pub const TYPE_METHOD: &str = "METHOD";
pub const TYPE_SOLUTION: &str = "SOLUTION";
pub const TYPE_VERIFICATION: &str = "VERIFICATION";
pub const TYPE_CONTRADICTION: &str = "CONTRADICTION";
pub const TYPE_ANSWER: &str = "ANSWER";
pub const TYPE_START: &str = "START";
pub const TYPE_QUANTITY: &str = "QUANTITY";
pub const TYPE_HYPOTHESIS: &str = "HYPOTHESIS";
pub const TYPE_OBSERVATION: &str = "OBSERVATION";
pub const TYPE_PLAN: &str = "PLAN";
pub const TYPE_TOOL_RESULT: &str = "TOOL_RESULT";
pub const TYPE_MEMORY: &str = "MEMORY";
pub const TYPE_CODE_BLOCK: &str = "CODE_BLOCK";

pub const EDGE_ENTAILS: &str = "entails";
pub const EDGE_CONTRADICTS: &str = "contradicts";
pub const EDGE_CAUSES: &str = "causes";
pub const EDGE_REQUIRES: &str = "requires";
pub const EDGE_VERIFIES: &str = "verifies";
pub const EDGE_PRECEDES: &str = "precedes";
pub const EDGE_SUPPORTS: &str = "supports";
pub const EDGE_REFERENCES: &str = "references";

fn valid_domains() -> HashSet<&'static str> {
    [
        DOMAIN_UNKNOWN,
        DOMAIN_CODE,
        DOMAIN_MATH,
        DOMAIN_LOGIC,
        DOMAIN_CHAT,
        DOMAIN_PHYSICS,
        DOMAIN_GENERAL,
    ]
    .into_iter()
    .collect()
}

fn valid_node_types() -> HashSet<&'static str> {
    [
        TYPE_UNKNOWN,
        TYPE_QUESTION,
        TYPE_FACT,
        TYPE_RULE,
        TYPE_ACTION,
        TYPE_RESPONSE,
        TYPE_ERROR,
        TYPE_FORMULA,
        TYPE_CONSTRAINT,
        TYPE_METHOD,
        TYPE_SOLUTION,
        TYPE_VERIFICATION,
        TYPE_CONTRADICTION,
        TYPE_ANSWER,
        TYPE_START,
        TYPE_QUANTITY,
        TYPE_HYPOTHESIS,
        TYPE_OBSERVATION,
        TYPE_PLAN,
        TYPE_TOOL_RESULT,
        TYPE_MEMORY,
        TYPE_CODE_BLOCK,
    ]
    .into_iter()
    .collect()
}

fn valid_edge_types() -> HashSet<&'static str> {
    [
        EDGE_ENTAILS,
        EDGE_CONTRADICTS,
        EDGE_CAUSES,
        EDGE_REQUIRES,
        EDGE_VERIFIES,
        EDGE_PRECEDES,
        EDGE_SUPPORTS,
        EDGE_REFERENCES,
    ]
    .into_iter()
    .collect()
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RinetoNode {
    #[pyo3(get)]
    pub id: u64,

    #[pyo3(get, set)]
    pub domain: String,

    #[pyo3(get, set)]
    pub node_type: String,

    #[pyo3(get, set)]
    pub content: String,

    #[pyo3(get, set)]
    pub confidence: f32,

    #[pyo3(get, set)]
    pub status: String,

    #[pyo3(get, set)]
    pub source: String,
}

#[pymethods]
impl RinetoNode {
    #[new]
    #[pyo3(signature = (
    id,
    domain,
    node_type,
    content,
    confidence=1.0,
    status="unknown".to_string(),
                        source="unknown".to_string()
    ))]
    fn new(
        id: u64,
        domain: String,
        node_type: String,
        content: String,
        confidence: f32,
        status: String,
        source: String,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(PyValueError::new_err(
                "confidence должна быть в диапазоне от 0.0 до 1.0",
            ));
        }

        Ok(Self {
            id,
            domain,
            node_type,
            content,
            confidence,
            status,
            source,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RinetoNode(id={}, domain='{}', node_type='{}', content='{}')",
                self.id, self.domain, self.node_type, self.content
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self)
        .map_err(|error| PyValueError::new_err(format!("Ошибка JSON: {error}")))
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RinetoEdge {
    #[pyo3(get)]
    pub source: u64,

    #[pyo3(get)]
    pub target: u64,

    #[pyo3(get, set)]
    pub edge_type: String,

    #[pyo3(get, set)]
    pub weight: f32,

    #[pyo3(get, set)]
    pub source_info: String,
}

#[pymethods]
impl RinetoEdge {
    #[new]
    #[pyo3(signature = (
    source,
    target,
    edge_type,
    weight=1.0,
    source_info="unknown".to_string()
    ))]
    fn new(
        source: u64,
        target: u64,
        edge_type: String,
        weight: f32,
        source_info: String,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(PyValueError::new_err(
                "weight должна быть в диапазоне от 0.0 до 1.0",
            ));
        }

        Ok(Self {
            source,
            target,
            edge_type,
            weight,
            source_info,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RinetoEdge({} -[{}]-> {})",
                self.source, self.edge_type, self.target
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<RinetoNode>,
    edges: Vec<RinetoEdge>,
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoGraph {
    nodes: Vec<RinetoNode>,
    edges: Vec<RinetoEdge>,
    next_id: u64,
}

#[pymethods]
impl RinetoGraph {
    #[new]
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 1,
        }
    }

    /// Добавляет узел и возвращает его ID.
    #[pyo3(signature = (
    domain,
    node_type,
    content,
    confidence=1.0,
    status="unknown".to_string(),
                        source="unknown".to_string()
    ))]
    fn add_node(
        &mut self,
        domain: String,
        node_type: String,
        content: String,
        confidence: f32,
        status: String,
        source: String,
    ) -> PyResult<u64> {
        self.validate_domain_name(&domain)?;
        self.validate_node_type_name(&node_type)?;

        let id = self.next_id;
        self.next_id += 1;

        let node = RinetoNode::new(
            id,
            domain,
            node_type,
            content,
            confidence,
            status,
            source,
        )?;

        self.nodes.push(node);
        Ok(id)
    }

    /// Добавляет связь между существующими узлами.
    #[pyo3(signature = (
    source,
    target,
    edge_type,
    weight=1.0,
    source_info="unknown".to_string()
    ))]
    fn add_edge(
        &mut self,
        source: u64,
        target: u64,
        edge_type: String,
        weight: f32,
        source_info: String,
    ) -> PyResult<()> {
        if source == target {
            return Err(PyValueError::new_err(
                "Нельзя создать связь узла с самим собой",
            ));
        }

        self.validate_edge_type_name(&edge_type)?;
        self.get_node(source)?;
        self.get_node(target)?;

        self.edges.push(RinetoEdge::new(
            source,
            target,
            edge_type,
            weight,
            source_info,
        )?);

        Ok(())
    }

        fn get_node(&self, node_id: u64) -> PyResult<RinetoNode> {
        self.nodes
        .iter()
        .find(|node| node.id == node_id)
        .cloned()
        .ok_or_else(|| PyKeyError::new_err(format!("Узел {node_id} не найден")))
    }

    fn nodes(&self) -> Vec<RinetoNode> {
        self.nodes.clone()
    }

    fn edges(&self) -> Vec<RinetoEdge> {
        self.edges.clone()
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Возвращает ошибки валидации.
    fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        let node_map: HashMap<u64, &RinetoNode> =
        self.nodes.iter().map(|node| (node.id, node)).collect();

        for node in &self.nodes {
            if !valid_domains().contains(node.domain.as_str()) {
                errors.push(format!(
                    "Узел {} содержит неизвестный домен '{}'",
                    node.id, node.domain
                ));
            }

            if !valid_node_types().contains(node.node_type.as_str()) {
                errors.push(format!(
                    "Узел {} содержит неизвестный тип '{}'",
                    node.id, node.node_type
                ));
            }

            if !(0.0..=1.0).contains(&node.confidence) {
                errors.push(format!(
                    "Узел {} имеет некорректную confidence: {}",
                    node.id, node.confidence
                ));
            }
        }

        for edge in &self.edges {
            let source = match node_map.get(&edge.source) {
                Some(node) => *node,
                None => {
                    errors.push(format!(
                        "Связь ссылается на отсутствующий source {}",
                        edge.source
                    ));
                    continue;
                }
            };

            let target = match node_map.get(&edge.target) {
                Some(node) => *node,
                None => {
                    errors.push(format!(
                        "Связь ссылается на отсутствующий target {}",
                        edge.target
                    ));
                    continue;
                }
            };

            if !valid_edge_types().contains(edge.edge_type.as_str()) {
                errors.push(format!(
                    "Неизвестный тип связи '{}'",
                    edge.edge_type
                ));
            }

            if !(0.0..=1.0).contains(&edge.weight) {
                errors.push(format!(
                    "Связь {} -> {} имеет некорректный weight",
                    edge.source, edge.target
                ));
            }

            if edge.edge_type == EDGE_VERIFIES
                && target.node_type != TYPE_SOLUTION
                && target.node_type != TYPE_ANSWER
                && target.node_type != TYPE_CODE_BLOCK
                {
                    errors.push(format!(
                        "Связь verifies должна указывать на SOLUTION, ANSWER или CODE_BLOCK: {} -> {}",
                        edge.source, edge.target
                    ));
                }

                if edge.edge_type == EDGE_REQUIRES
                    && source.node_type == TYPE_ANSWER
                    {
                        errors.push(format!(
                            "ANSWER не должен требовать другой узел: {} -> {}",
                            edge.source, edge.target
                        ));
                    }

                    if edge.edge_type == EDGE_CONTRADICTS
                        && source.node_type != TYPE_CONTRADICTION
                        && target.node_type != TYPE_CONTRADICTION
                        {
                            errors.push(format!(
                                "Для contradicts один из узлов должен иметь тип CONTRADICTION: {} -> {}",
                                edge.source, edge.target
                            ));
                        }

                        if edge.edge_type == EDGE_PRECEDES
                            && source.node_type == TYPE_ANSWER
                            {
                                errors.push(format!(
                                    "ANSWER не может precede другой узел: {} -> {}",
                                    edge.source, edge.target
                                ));
                            }
        }

        let has_solution = self.nodes.iter().any(|n| n.node_type == TYPE_SOLUTION);
        let has_answer = self.nodes.iter().any(|n| n.node_type == TYPE_ANSWER);

        if has_answer && !has_solution {
            let has_response = self.nodes.iter().any(|n| n.node_type == TYPE_RESPONSE);

            if !has_response {
                errors.push(
                    "ANSWER должен иметь SOLUTION или RESPONSE в графе".to_string(),
                );
            }
        }

        // Проверка обязательной связи VERIFICATION -> SOLUTION/ANSWER/CODE_BLOCK.
        for node in &self.nodes {
            if node.node_type == TYPE_VERIFICATION {
                let verifies_something = self
                .edges
                .iter()
                .any(|edge| edge.source == node.id && edge.edge_type == EDGE_VERIFIES);

                if !verifies_something {
                    errors.push(format!(
                        "Узел VERIFICATION {} не проверяет ни одного результата",
                        node.id
                    ));
                }
            }
        }

        errors
    }

    fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    fn to_json(&self) -> PyResult<String> {
        let data = GraphData {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
        };

        serde_json::to_string_pretty(&data)
        .map_err(|error| PyValueError::new_err(format!("Ошибка сериализации: {error}")))
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        let data: GraphData = serde_json::from_str(&json)
        .map_err(|error| PyValueError::new_err(format!("Некорректный JSON графа: {error}")))?;

        let next_id = data
        .nodes
        .iter()
        .map(|node| node.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

        Ok(Self {
            nodes: data.nodes,
            edges: data.edges,
            next_id,
        })
    }

    #[pyo3(signature = (question, answer, domain="GENERAL".to_string()))]
    fn reason(
        &mut self,
        question: String,
        answer: String,
        domain: String,
    ) -> PyResult<Vec<String>> {
        self.validate_domain_name(&domain)?;

        // 1. Создаём узел ВОПРОСА
        let q_id = self.add_node(
            domain.clone(),
                                 TYPE_QUESTION.to_string(),
                                 question.clone(),
                                 1.0,
                                 "open".to_string(),
                                 "user".to_string(),
        )?;

        // 2. КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Создаём узел РЕШЕНИЯ (SOLUTION)
        // Это удовлетворяет инвариант валидатора: ANSWER должен иметь SOLUTION.
        let s_id = self.add_node(
            domain.clone(),
                                 TYPE_SOLUTION.to_string(),
                                 answer.clone(),
                                 1.0,
                                 "proposed".to_string(),
                                 "rineto".to_string(),
        )?;

        // 3. Создаём узел ОТВЕТА
        let a_id = self.add_node(
            domain.clone(),
                                 TYPE_ANSWER.to_string(),
                                 answer.clone(),
                                 1.0,
                                 "verified".to_string(),
                                 "rineto".to_string(),
        )?;

        let mut verifier = RinetoZ3Verifier::new();
        let mut steps: Vec<String> = Vec::new();

        // Связываем QUESTION -> SOLUTION (а не QUESTION -> ANSWER)
        if verifier.verify_entails(question.clone(), answer.clone())? {
            self.add_edge(q_id, s_id, EDGE_ENTAILS.to_string(), 1.0, "verifier".to_string())?;
            steps.push(format!("entails: {question} -> {answer}"));
        }

        if verifier.verify_verifies(question.clone(), answer.clone())? {
            let v_id = self.add_node(
                domain.clone(),
                                     TYPE_VERIFICATION.to_string(),
                                     format!("Проверка: {question}"),
                                         1.0,
                                     "done".to_string(),
                                     "verifier".to_string(),
            )?;
            self.add_edge(v_id, s_id, EDGE_VERIFIES.to_string(), 1.0, "verifier".to_string())?;
            steps.push(format!("verifies: {question} проверяет {answer}"));
        }

        if verifier.verify_requires(question.clone(), answer.clone())? {
            self.add_edge(q_id, s_id, EDGE_REQUIRES.to_string(), 0.8, "verifier".to_string())?;
            steps.push(format!("requires: {answer} требует {question}"));
        }

        if verifier.verify_precedes(question.clone(), answer.clone())? {
            self.add_edge(q_id, s_id, EDGE_PRECEDES.to_string(), 0.8, "verifier".to_string())?;
            steps.push(format!("precedes: {question} перед {answer}"));
        }

        if verifier.verify_causes(question.clone(), answer.clone())? {
            self.add_edge(q_id, s_id, EDGE_CAUSES.to_string(), 0.7, "verifier".to_string())?;
            steps.push(format!("causes: {question} вызывает {answer}"));
        }

        if verifier.verify_contradicts(question.clone(), answer.clone())? {
            let c_id = self.add_node(
                domain.clone(),
                                     TYPE_CONTRADICTION.to_string(),
                                     format!("Противоречие: {question} <-> {answer}"),
                                         1.0,
                                     "open".to_string(),
                                     "verifier".to_string(),
            )?;
            self.add_edge(q_id, c_id, EDGE_CONTRADICTS.to_string(), 1.0, "verifier".to_string())?;
            self.add_edge(a_id, c_id, EDGE_CONTRADICTS.to_string(), 1.0, "verifier".to_string())?;
            steps.push(format!("contradicts: {question} противоречит {answer}"));
        }

        Ok(steps)
    }

    fn __repr__(&self) -> String {
        format!(
            "RinetoGraph(nodes={}, edges={}, valid={})",
                self.nodes.len(),
                self.edges.len(),
                self.is_valid()
        )
    }
}

impl RinetoGraph {
    fn validate_domain_name(&self, domain: &str) -> PyResult<()> {
        if !valid_domains().contains(domain) {
            return Err(PyValueError::new_err(format!(
                "Неизвестный домен '{domain}'"
            )));
        }

        Ok(())
    }

    fn validate_node_type_name(&self, node_type: &str) -> PyResult<()> {
        if !valid_node_types().contains(node_type) {
            return Err(PyValueError::new_err(format!(
                "Неизвестный тип узла '{node_type}'"
            )));
        }

        Ok(())
    }

    fn validate_edge_type_name(&self, edge_type: &str) -> PyResult<()> {
        if !valid_edge_types().contains(edge_type) {
            return Err(PyValueError::new_err(format!(
                "Неизвестный тип связи '{edge_type}'"
            )));
        }

        Ok(())
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RinetoTemplate {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub domain: String,

    #[pyo3(get)]
    pub input_types: Vec<String>,

    #[pyo3(get)]
    pub output_types: Vec<String>,

    #[pyo3(get)]
    pub tools: Vec<String>,

    #[pyo3(get)]
    pub required_verification: bool,
}

#[pymethods]
impl RinetoTemplate {
    #[new]
    #[pyo3(signature = (
    name,
    domain,
    input_types,
    output_types,
    tools=Vec::new(),
                        required_verification=false
    ))]
    fn new(
        name: String,
        domain: String,
        input_types: Vec<String>,
        output_types: Vec<String>,
        tools: Vec<String>,
        required_verification: bool,
    ) -> PyResult<Self> {
        if !valid_domains().contains(domain.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Неизвестный домен шаблона '{domain}'"
            )));
        }

        Ok(Self {
            name,
            domain,
            input_types,
            output_types,
            tools,
            required_verification,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RinetoTemplate(name='{}', domain='{}')",
                self.name, self.domain
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct RouteDecision {
    #[pyo3(get)]
    pub domain: String,

    #[pyo3(get)]
    pub expert: String,

    #[pyo3(get)]
    pub template: String,

    #[pyo3(get)]
    pub tools: Vec<String>,

    #[pyo3(get)]
    pub reason: String,

    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl RouteDecision {
    fn to_dict(&self) -> HashMap<String, String> {
        let mut result = HashMap::new();

        result.insert("domain".to_string(), self.domain.clone());
        result.insert("expert".to_string(), self.expert.clone());
        result.insert("template".to_string(), self.template.clone());
        result.insert("tools".to_string(), self.tools.join(","));
        result.insert("reason".to_string(), self.reason.clone());
        result.insert("confidence".to_string(), self.confidence.to_string());

        result
    }

    fn __repr__(&self) -> String {
        format!(
            "RouteDecision(domain='{}', expert='{}', template='{}', tools={:?}, confidence={})",
                self.domain,
                self.expert,
                self.template,
                self.tools,
                self.confidence
        )
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoRouter;

/// Результат исполнения маршрута RinetoPipeline.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PipelineResult {
    #[pyo3(get)]
    pub query: String,

    #[pyo3(get)]
    pub domain: String,

    #[pyo3(get)]
    pub expert: String,

    #[pyo3(get)]
    pub template: String,

    #[pyo3(get)]
    pub tools: Vec<String>,

    #[pyo3(get)]
    pub reason: String,

    #[pyo3(get)]
    pub confidence: f32,

    #[pyo3(get)]
    pub route_valid: bool,

    #[pyo3(get)]
    pub verified: bool,

    #[pyo3(get)]
    pub steps: Vec<String>,
}

/// Оркестратор Ринэто: исполняет маршрут целиком.
///
/// `run(query)` выполняет цепочку:
/// Router → RinetoASM (построение и валидация маршрута) → Z3Verifier (проверка)
/// и возвращает структурированный результат. Контекст диалога накапливается
/// в `RinetoContext`, а подтверждённые маршруты сохраняются в векторной памяти
/// и уточняют последующие запросы.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoPipeline {
    router: RinetoRouter,
    verifier: RinetoZ3Verifier,
    context: RinetoContext,
    route_memory: RinetoVectorMemory,

    #[pyo3(get)]
    pub total_queries: u64,

    #[pyo3(get)]
    pub verified_queries: u64,

    #[pyo3(get)]
    pub failed_routes: u64,

    #[pyo3(get)]
    pub memory_hits: u64,
}

#[pymethods]
impl RinetoPipeline {
    #[new]
    #[pyo3(signature = (context_alpha=0.999, memory_dim=64))]
    fn new(context_alpha: f32, memory_dim: usize) -> PyResult<Self> {
        if memory_dim == 0 {
            return Err(PyValueError::new_err(
                "Размерность памяти маршрутов должна быть больше нуля",
            ));
        }
        Ok(Self {
            router: RinetoRouter,
            verifier: RinetoZ3Verifier::new(),
            context: RinetoContext::new(context_alpha)?,
            route_memory: RinetoVectorMemory::new(),
            total_queries: 0,
            verified_queries: 0,
            failed_routes: 0,
            memory_hits: 0,
        })
    }

    /// Исполняет маршрут для запроса.
    fn run(&mut self, query: String) -> PyResult<PipelineResult> {
        self.total_queries += 1;

        // Контекст накапливает историю.
        self.context.update(query.clone());

        // Память маршрутов: если похожий запрос уже обработан с доменом выше
        // порога, считаем попадание и используем сохранённый домен.
        let query_vec = query_signature_vector(&query, self.route_memory_dim());
        let mut domain_override: Option<String> = None;
        if let Some((_, _, distance, label)) = self
            .route_memory
            .search_exact(RinetoVector::new(query_vec.clone(), 1.0)?, 1)?
            .into_iter()
            .next()
        {
            if distance < ROUTE_MEMORY_SIMILARITY_THRESHOLD {
                self.memory_hits += 1;
                domain_override = Some(label);
            }
        }

        let decision = self.router.route(query.clone());
        let decision = if let Some(domain) = domain_override {
            if decision.domain != domain {
                RouteDecision {
                    domain: domain.clone(),
                    expert: decision.expert.clone(),
                    template: decision.template.clone(),
                    tools: decision.tools.clone(),
                    reason: format!(
                        "Память маршрутов: {} (было {})",
                        domain, decision.domain
                    ),
                    confidence: decision.confidence.max(0.5),
                }
            } else {
                decision
            }
        } else {
            decision
        };

        // Строим маршрут ASM из решения маршрутизатора.
        let steps = match build_pipeline_steps(&decision) {
            Some(steps) => steps,
            None => {
                self.failed_routes += 1;
                return Ok(PipelineResult {
                    query,
                    domain: decision.domain,
                    expert: decision.expert,
                    template: decision.template,
                    tools: decision.tools,
                    reason: decision.reason,
                    confidence: decision.confidence,
                    route_valid: false,
                    verified: false,
                    steps: Vec::new(),
                });
            }
        };

        // Проверяем маршрут через статическую валидацию ASM.
        let mut asm = RinetoASM::new();
        let route_valid = asm.assemble(steps.join("\n")).is_ok();

        // Проверка сходимости: уверенность достаточно высока?
        let verified = route_valid
            && decision.confidence >= VERIFICATION_CONFIDENCE_THRESHOLD
            && self.verifier.verify_entails(query.clone(), query.clone())?;
        if verified {
            self.verified_queries += 1;
            // Сохраняем подтверждённый маршрут в память.
            let vector = RinetoVector::new(query_vec.clone(), 1.0)?;
            self.route_memory.add(vector, decision.domain.clone());
        }

        Ok(PipelineResult {
            query,
            domain: decision.domain,
            expert: decision.expert,
            template: decision.template,
            tools: decision.tools,
            reason: decision.reason,
            confidence: decision.confidence,
            route_valid,
            verified,
            steps,
        })
    }

    /// Возвращает сжатое состояние диалога (вектор контекста).
    fn context_state(&self) -> Vec<f32> {
        self.context.state()
    }

    fn context_updates(&self) -> u64 {
        self.context.updates
    }

    fn reset_context(&mut self) {
        self.context.reset();
    }

    fn memory_size(&self) -> usize {
        self.route_memory.len()
    }

    fn clear_memory(&mut self) {
        self.route_memory.clear();
    }
}

impl RinetoPipeline {
    fn route_memory_dim(&self) -> usize {
        64
    }
}

const ROUTE_MEMORY_SIMILARITY_THRESHOLD: f32 = 0.5;

/// Компактный вектор-подпись текста для памяти маршрутов.
fn query_signature_vector(text: &str, dim: usize) -> Vec<f32> {
    let digest = fractal_compress(text);
    let mut vector = vec![0.0f32; dim];
    for (index, value) in digest.iter().enumerate() {
        if index < dim {
            vector[index] = *value;
        }
    }
    vector
}

const VERIFICATION_CONFIDENCE_THRESHOLD: f32 = 0.5;

fn build_pipeline_steps(decision: &RouteDecision) -> Option<Vec<String>> {
    let mut steps = vec![format!("CLASSIFY_DOMAIN {}", decision.domain)];
    steps.push(format!("SELECT_EXPERT {}", decision.expert));
    steps.push(format!("SELECT_TEMPLATE {}", decision.template));
    for tool in &decision.tools {
        steps.push(format!("CALL_TOOL {tool}"));
    }
    steps.push("VERIFY".to_string());
    steps.push("RETURN".to_string());
    Some(steps)
}

const MICRON_PATCH_PIXELS: usize = 16 * 16;
const MICRON_SIGNATURE_WORDS: usize = 4;

/// Бинарный дескриптор патча 16x16 для быстрого поиска по Хэммингу.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicronPatch {
    #[pyo3(get)]
    pub signature: [u64; MICRON_SIGNATURE_WORDS],

    #[pyo3(get)]
    pub mean_value: f32,

    #[pyo3(get)]
    pub position: (u16, u16),
}

#[pymethods]
impl MicronPatch {
    #[new]
    #[pyo3(signature = (pixels, position=(0, 0)))]
    fn new(pixels: Vec<f32>, position: (u16, u16)) -> PyResult<Self> {
        let (signature, mean_value) = micron_signature(&pixels)?;

        Ok(Self {
            signature,
            mean_value,
            position,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MicronPatch(mean_value={:.4}, position=({},{}))",
            self.mean_value, self.position.0, self.position.1
        )
    }
}

/// Запись визуальной ассоциативной памяти Микрона.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicronMemoryItem {
    #[pyo3(get)]
    pub patch: MicronPatch,

    #[pyo3(get)]
    pub label: String,
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct MicronMemory {
    items: Vec<MicronMemoryItem>,
}

/// Float-вектор с бинарной подписью для быстрого предварительного поиска.
///
/// Подпись использует первые 64 компонента, а float-данные сохраняются для
/// точного сравнения и последующего обучения.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RinetoVector {
    #[pyo3(get)]
    pub data: Vec<f32>,

    #[pyo3(get)]
    pub signature: u64,

    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get, set)]
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RinetoVectorMemoryItem {
    vector: RinetoVector,
    label: String,
    /// Предвычисленная ||v||² для быстрого L2 через dot (приём sklearn).
    norm_sq: f32,
}

impl RinetoVectorMemoryItem {
    fn new(vector: RinetoVector, label: String) -> Self {
        let norm_sq: f32 = vector.data.iter().map(|v| v * v).sum();
        Self {
            vector,
            label,
            norm_sq,
        }
    }
}

/// L2² между запросом и эталоном через предвычисленную норму:
/// dist² = q_norm² + e_norm² − 2·(q·e). (паттерн sklearn euclidean_distances)
fn l2_sq_via_dot(query: &[f32], query_norm_sq: f32, item: &RinetoVectorMemoryItem) -> f32 {
    let dot = dot_simd(query, &item.vector.data);
    let dist_sq = query_norm_sq + item.norm_sq - 2.0 * dot;
    dist_sq.max(0.0)
}

/// Ассоциативная память векторов: быстрый бинарный фильтр и точное L2-уточнение.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoVectorMemory {
    items: Vec<RinetoVectorMemoryItem>,
}

/// Обучаемая матрица эталонов Ринэто.
///
/// Поиск выполняется в две фазы: бинарный Hamming-фильтр, затем точное
/// сравнение float-векторов среди ограниченного числа кандидатов.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoMatrix {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub total_ops: u64,

    #[pyo3(get)]
    pub updates: u64,

    #[pyo3(get)]
    pub cache_hits: u64,

    #[pyo3(get)]
    pub cache_misses: u64,

    entries: Vec<RinetoVectorMemoryItem>,
    cache: Vec<Option<(u64, Vec<(usize, u32, f32, String)>)>>,
}

const MATRIX_CACHE_SIZE: usize = 256;

#[pymethods]
impl RinetoMatrix {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("Размерность матрицы должна быть больше нуля"));
        }

        Ok(Self {
            dim,
            total_ops: 0,
            updates: 0,
            cache_hits: 0,
            cache_misses: 0,
            entries: Vec::new(),
            cache: vec![None; MATRIX_CACHE_SIZE],
        })
    }

    /// Добавляет обучающий эталон в матрицу.
    fn add(&mut self, data: Vec<f32>, label: String) -> PyResult<()> {
        self.ensure_dimension(data.len())?;
        self.entries.push(RinetoVectorMemoryItem::new(
            RinetoVector::new(data, 1.0)?,
            label,
        ));
        self.invalidate_cache();
        Ok(())
    }

    /// Кэш результатов (сверхпроводящий режим из RetoTensor):
    /// повторный запрос с той же сигнатурой возвращает результат за O(1).
    fn check_cache(&self, signature: u64, k: usize) -> Option<Vec<(usize, u32, f32, String)>> {
        let slot = (signature as usize) % MATRIX_CACHE_SIZE;
        if let Some((cached_sig, cached_result)) = &self.cache[slot] {
            if *cached_sig == signature && cached_result.len() >= k.min(cached_result.len()) {
                return Some(cached_result.clone());
            }
        }
        None
    }

    fn store_cache(
        &mut self,
        signature: u64,
        result: Vec<(usize, u32, f32, String)>,
    ) {
        let slot = (signature as usize) % MATRIX_CACHE_SIZE;
        self.cache[slot] = Some((signature, result));
    }

    fn invalidate_cache(&mut self) {
        for slot in self.cache.iter_mut() {
            *slot = None;
        }
    }

    fn get_cache_stats(&self) -> (u64, u64, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }

    /// Двухфазный поиск ближайших эталонов с multi-probe фильтром.
    /// Возвращает `(index, hamming, l2, label)`.
    #[pyo3(signature = (query, k=1, candidate_multiplier=4, probes=2))]
    fn forward(
        &mut self,
        query: Vec<f32>,
        k: usize,
        candidate_multiplier: usize,
        probes: usize,
    ) -> PyResult<Vec<(usize, u32, f32, String)>> {
        self.ensure_dimension(query.len())?;
        if k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let query_vector = RinetoVector::new(query, 1.0)?;
        self.total_ops += 1;

        // Сверхпроводящий режим: повторный запрос из кэша.
        if let Some(cached) = self.check_cache(query_vector.signature, k) {
            self.cache_hits += 1;
            return Ok(cached);
        }
        self.cache_misses += 1;

        let candidates = hamming_probe_candidates(
            &query_vector,
            &self.entries,
            probes,
            k.saturating_mul(candidate_multiplier.max(1)).max(k),
        );

        let mut results = candidates
            .into_iter()
            .map(|(index, hamming)| {
                let entry = &self.entries[index];
                Ok((
                    index,
                    hamming,
                    l2_distance_fast(&query_vector.data, &entry.vector.data),
                    entry.label.clone(),
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        results.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(results.len()));

        // Сохраняем в кэш только результат с максимальным k (перезаписываем).
        if results.len() >= k && k > 0 {
            self.store_cache(query_vector.signature, results.clone());
        }

        Ok(results)
    }

    /// Точный поиск: полный L2 по всем эталонам. recall@1 = 1.0.
    #[pyo3(signature = (query, k=1))]
    fn forward_exact(
        &mut self,
        query: Vec<f32>,
        k: usize,
    ) -> PyResult<Vec<(usize, u32, f32, String)>> {
        self.ensure_dimension(query.len())?;
        if k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let query_vector = RinetoVector::new(query, 1.0)?;
        let query_norm_sq: f32 = query_vector.data.iter().map(|v| v * v).sum();
        self.total_ops += 1;

        let mut results: Vec<(usize, u32, f32, String)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    index,
                    (query_vector.signature ^ entry.vector.signature).count_ones(),
                    l2_sq_via_dot(&query_vector.data, query_norm_sq, entry).sqrt(),
                    entry.label.clone(),
                )
            })
            .collect();
        results.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(results.len()));
        Ok(results)
    }

    /// Реконструктор: предсказывает следующий токен, исключая текущий (exclude_idx),
    /// чтобы избежать зацикливания на себе. Возвращает (index, confidence).
    #[pyo3(signature = (query, exclude_idx, k=1))]
    fn predict_next(
        &mut self,
        query: Vec<f32>,
        exclude_idx: usize,
        k: usize,
    ) -> PyResult<Vec<(usize, u32, f32, String)>> {
        self.ensure_dimension(query.len())?;
        if k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let query_vector = RinetoVector::new(query, 1.0)?;
        self.total_ops += 1;

        let mut candidates: Vec<(usize, u32)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != exclude_idx)
            .map(|(index, entry)| {
                (
                    index,
                    (query_vector.signature ^ entry.vector.signature).count_ones(),
                )
            })
            .collect();
        candidates.sort_unstable_by_key(|candidate| candidate.1);
        let take = k.min(candidates.len());

        let mut results = candidates
            .into_iter()
            .take(take)
            .map(|(index, hamming)| {
                let entry = &self.entries[index];
                Ok((
                    index,
                    hamming,
                    l2_distance_fast(&query_vector.data, &entry.vector.data),
                    entry.label.clone(),
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        results.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Пакетный двухфазный поиск для массива запросов.
    #[pyo3(signature = (queries, k=1, candidate_multiplier=4, probes=2, exact=false))]
    fn forward_batch(
        &mut self,
        queries: Vec<Vec<f32>>,
        k: usize,
        candidate_multiplier: usize,
        probes: usize,
        exact: bool,
    ) -> PyResult<Vec<Vec<(usize, u32, f32, String)>>> {
        let mut batch = Vec::with_capacity(queries.len());
        for query in queries {
            if exact {
                batch.push(self.forward_exact(query, k)?);
            } else {
                batch.push(self.forward(query, k, candidate_multiplier, probes)?);
            }
        }
        Ok(batch)
    }

    /// Лёгкий точный пакетный поиск для обучения: только `(index, l2)` без меток.
    /// Минимизирует накладные расходы на кортежи со строками.
    #[pyo3(signature = (queries, k=1))]
    fn forward_exact_indices_batch(
        &mut self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let mut batch = Vec::with_capacity(queries.len());
        for query in queries {
            self.ensure_dimension(query.len())?;
            if k == 0 || self.entries.is_empty() {
                batch.push(Vec::new());
                continue;
            }
            self.total_ops += 1;

            let query_norm_sq: f32 = query.iter().map(|v| v * v).sum();
            let mut results: Vec<(usize, f32)> = self
                .entries
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    (
                        index,
                        l2_sq_via_dot(&query, query_norm_sq, entry).sqrt(),
                    )
                })
                .collect();
            results.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k.min(results.len()));
            batch.push(results);
        }
        Ok(batch)
    }

    /// Обновляет ближайший эталон по правилу EMA и возвращает loss до обновления.
    #[pyo3(signature = (query, target, learning_rate=0.1))]
    fn train_step(
        &mut self,
        query: Vec<f32>,
        target: Vec<f32>,
        learning_rate: f32,
    ) -> PyResult<f32> {
        self.ensure_dimension(query.len())?;
        self.ensure_dimension(target.len())?;
        if self.entries.is_empty() {
            self.add(target, "learned".to_string())?;
            self.updates += 1;
            return Ok(0.0);
        }

        let nearest = self.forward(query, 1, 4, 2)?;
        let index = nearest[0].0;
        let target = RinetoVector::new(target, 1.0)?;
        let entry = &mut self.entries[index].vector;
        let loss = entry.l2_distance(&target)?.powi(2);
        entry.learn_towards(&target, learning_rate)?;
        self.updates += 1;
        Ok(loss)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

impl RinetoMatrix {
    fn ensure_dimension(&self, actual: usize) -> PyResult<()> {
        if actual != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается размерность {}, получено {}",
                self.dim, actual
            )));
        }
        Ok(())
    }
}

fn l2_distance_fast(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_distance_avx2(left, right) };
        }
    }
    let mut sum = 0.0f32;
    for (a, b) in left.iter().zip(right) {
        let diff = a - b;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// AVX2+FMA-версия L2-расстояния: y = sqrt(sum((a-b)^2)).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        let offset = c * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
        + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    let base = chunks * 8;
    for i in 0..remainder {
        let diff = a[base + i] - b[base + i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// AVX2-ускорение скалярного произведения: sum(a[i]*b[i]).
#[inline(always)]
pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        let offset = c * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
        + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    let base = chunks * 8;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }
    sum
}

/// Multi-probe Hamming-фильтр: генерирует probe-подписи, переворачивая биты
/// самых неопределённых (близких к нулю) компонентов запроса, и возвращает
/// лучших кандидатов по минимальному Hamming-расстоянию среди всех probe.
///
/// Собирает кандидатов в один Vec с последующей сортировкой и дедупликацией —
/// без per-query аллокаций HashMap.
fn hamming_probe_candidates(
    query: &RinetoVector,
    entries: &[RinetoVectorMemoryItem],
    probes: usize,
    budget: usize,
) -> Vec<(usize, u32)> {
    if entries.is_empty() || budget == 0 {
        return Vec::new();
    }

    let n_probes = probes.min(64).min(query.data.len());

    if n_probes == 0 {
        // Быстрый путь: один проход с сортировкой.
        let mut candidates: Vec<(usize, u32)> = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    index,
                    (query.signature ^ entry.vector.signature).count_ones(),
                )
            })
            .collect();
        candidates.sort_unstable_by_key(|candidate| candidate.1);
        candidates.truncate(budget.min(candidates.len()));
        return candidates;
    }

    // Индексы компонентов, отсортированные по |значение| (самые близкие к 0
    // — самые неопределённые для бинаризации).
    let mut uncertain: Vec<(usize, f32)> = query
        .data
        .iter()
        .take(64)
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .collect();
    uncertain.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let probe_count = (n_probes + 1) * entries.len();
    let mut all: Vec<(usize, u32)> = Vec::with_capacity(probe_count);
    let mut probe_sig = query.signature;

    for probe in 0..=n_probes {
        if probe > 0 {
            if let Some(&(index, _)) = uncertain.get(probe - 1) {
                probe_sig ^= 1_u64 << index;
            }
        }
        for (entry_index, entry) in entries.iter().enumerate() {
            let hamming = (probe_sig ^ entry.vector.signature).count_ones();
            all.push((entry_index, hamming));
        }
    }

    all.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.cmp(&b.1))
    });

    // Дедупликация: оставляем минимальное hamming для каждого индекса.
    let mut candidates: Vec<(usize, u32)> = Vec::with_capacity(entries.len());
    let mut current_index = usize::MAX;
    for (index, hamming) in all {
        if index == current_index {
            continue;
        }
        current_index = index;
        candidates.push((index, hamming));
    }

    candidates.sort_unstable_by_key(|candidate| candidate.1);
    candidates.truncate(budget.min(candidates.len()));
    candidates
}

#[pymethods]
impl RinetoVectorMemory {
    #[new]
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn add(&mut self, vector: RinetoVector, label: String) {
        self.items.push(RinetoVectorMemoryItem::new(vector, label));
    }

    /// Возвращает `(index, hamming_distance, l2_distance, label)`.
    #[pyo3(signature = (query, k=1, probes=2))]
    fn search(&self, query: RinetoVector, k: usize, probes: usize) -> PyResult<Vec<(usize, u32, f32, String)>> {
        if self.items.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Проверка размерности: hamming_probe_candidates не знает о dim.
        if let Some(first) = self.items.first() {
            if query.dim != first.vector.dim {
                return Err(PyValueError::new_err(format!(
                    "Размерности векторов не совпадают: {} и {}",
                    query.dim, first.vector.dim
                )));
            }
        }

        let candidates = hamming_probe_candidates(
            &query,
            &self.items,
            probes,
            k.saturating_mul(4).max(k),
        );

        let query_norm_sq: f32 = query.data.iter().map(|v| v * v).sum();
        let mut results = candidates
            .into_iter()
            .map(|(index, hamming)| {
                let item = &self.items[index];
                Ok((
                    index,
                    hamming,
                    l2_sq_via_dot(&query.data, query_norm_sq, item).sqrt(),
                    item.label.clone(),
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;

        results.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(results.len()));
        Ok(results)
    }

    /// Точный L2-поиск по всем записям.
    #[pyo3(signature = (query, k=1))]
    fn search_exact(&self, query: RinetoVector, k: usize) -> PyResult<Vec<(usize, u32, f32, String)>> {
        if self.items.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let query_norm_sq: f32 = query.data.iter().map(|v| v * v).sum();
        let mut results: Vec<(usize, u32, f32, String)> = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                (
                    index,
                    (query.signature ^ item.vector.signature).count_ones(),
                    l2_sq_via_dot(&query.data, query_norm_sq, item).sqrt(),
                    item.label.clone(),
                )
            })
            .collect();

        results.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(results.len()));
        Ok(results)
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

#[pymethods]
impl RinetoVector {
    #[new]
    #[pyo3(signature = (data, confidence=1.0))]
    fn new(data: Vec<f32>, confidence: f32) -> PyResult<Self> {
        validate_vector(&data)?;
        validate_unit_value(confidence, "confidence")?;

        Ok(Self {
            dim: data.len(),
            signature: vector_signature(&data),
            data,
            confidence,
        })
    }

    /// Пересчитывает подпись после внешнего изменения `data`.
    fn binarize(&mut self) -> u64 {
        self.signature = vector_signature(&self.data);
        self.signature
    }

    fn hamming_distance(&self, other: &RinetoVector) -> PyResult<u32> {
        ensure_same_dimension(self, other)?;
        Ok((self.signature ^ other.signature).count_ones())
    }

    fn semantic_similarity(&self, other: &RinetoVector) -> PyResult<f32> {
        let distance = self.hamming_distance(other)?;
        Ok((64 - distance) as f32 / 64.0)
    }

    /// Точное L2-расстояние по float-представлениям.
    fn l2_distance(&self, other: &RinetoVector) -> PyResult<f32> {
        ensure_same_dimension(self, other)?;
        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f32>()
            .sqrt())
    }

    fn dot(&self, other: &RinetoVector) -> PyResult<f32> {
        ensure_same_dimension(self, other)?;
        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(left, right)| left * right)
            .sum())
    }

    fn normalize(&mut self) -> PyResult<f32> {
        let norm = self.data.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return Err(PyValueError::new_err(
                "Нельзя нормализовать нулевой вектор",
            ));
        }

        for value in &mut self.data {
            *value /= norm;
        }
        self.binarize();
        Ok(norm)
    }

    /// Один контролируемый шаг к целевому float-вектору.
    fn learn_towards(&mut self, target: &RinetoVector, learning_rate: f32) -> PyResult<f32> {
        ensure_same_dimension(self, target)?;
        validate_unit_value(learning_rate, "learning_rate")?;

        let mut loss = 0.0;
        for (value, target_value) in self.data.iter_mut().zip(&target.data) {
            let difference = *value - target_value;
            loss += difference * difference;
            *value -= learning_rate * difference;
        }
        self.binarize();
        Ok(loss)
    }

    fn __repr__(&self) -> String {
        format!("RinetoVector(dim={}, signature={:#018x})", self.dim, self.signature)
    }
}

fn validate_vector(data: &[f32]) -> PyResult<()> {
    if data.is_empty() {
        return Err(PyValueError::new_err("Вектор не может быть пустым"));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "Компоненты вектора должны быть конечными числами",
        ));
    }
    Ok(())
}

fn validate_unit_value(value: f32, name: &str) -> PyResult<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "{name} должен быть в диапазоне от 0.0 до 1.0"
        )));
    }
    Ok(())
}

fn vector_signature(data: &[f32]) -> u64 {
    data.iter().take(64).enumerate().fold(0_u64, |bits, (index, value)| {
        if *value > 0.0 {
            bits | (1_u64 << index)
        } else {
            bits
        }
    })
}

fn ensure_same_dimension(left: &RinetoVector, right: &RinetoVector) -> PyResult<()> {
    if left.dim != right.dim {
        return Err(PyValueError::new_err(format!(
            "Размерности векторов не совпадают: {} и {}",
            left.dim, right.dim
        )));
    }
    Ok(())
}

#[pymethods]
impl MicronMemory {
    #[new]
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Сохраняет патч и связанную с ним метку.
    fn add(&mut self, patch: MicronPatch, label: String) {
        self.items.push(MicronMemoryItem { patch, label });
    }

    /// Ищет ближайшие эталоны и возвращает (индекс, расстояние, метку).
    #[pyo3(signature = (patch, k=1))]
    fn search(&self, patch: MicronPatch, k: usize) -> Vec<(usize, u32, String)> {
        let mut matches: Vec<(usize, u32, String)> = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                (
                    index,
                    micron_hamming_distance(&patch.signature, &item.patch.signature),
                    item.label.clone(),
                )
            })
            .collect();

        matches.sort_unstable_by_key(|item| item.1);
        matches.truncate(k);
        matches
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

fn micron_signature(pixels: &[f32]) -> PyResult<([u64; MICRON_SIGNATURE_WORDS], f32)> {
    if pixels.len() != MICRON_PATCH_PIXELS {
        return Err(PyValueError::new_err(format!(
            "Микрон ожидает патч 16x16 из {} значений, получено {}",
            MICRON_PATCH_PIXELS,
            pixels.len()
        )));
    }

    if pixels.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "Пиксели патча должны быть конечными числами",
        ));
    }

    let mean_value = pixels.iter().sum::<f32>() / pixels.len() as f32;
    let mut signature = [0_u64; MICRON_SIGNATURE_WORDS];

    for (index, value) in pixels.iter().enumerate() {
        if *value > mean_value {
            signature[index / 64] |= 1_u64 << (index % 64);
        }
    }

    Ok((signature, mean_value))
}

fn micron_hamming_distance(
    a: &[u64; MICRON_SIGNATURE_WORDS],
    b: &[u64; MICRON_SIGNATURE_WORDS],
) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

#[pyfunction]
#[pyo3(signature = (pixels))]
fn micron_binarize(pixels: Vec<f32>) -> PyResult<(Vec<u64>, f32)> {
    let (signature, mean_value) = micron_signature(&pixels)?;
    Ok((signature.to_vec(), mean_value))
}

#[pyfunction]
fn micron_distance(a: Vec<u64>, b: Vec<u64>) -> PyResult<u32> {
    if a.len() != MICRON_SIGNATURE_WORDS || b.len() != MICRON_SIGNATURE_WORDS {
        return Err(PyValueError::new_err(
            "Подпись Микрона должна содержать ровно 4 числа u64",
        ));
    }

    let left: [u64; MICRON_SIGNATURE_WORDS] = a.try_into().unwrap();
    let right: [u64; MICRON_SIGNATURE_WORDS] = b.try_into().unwrap();
    Ok(micron_hamming_distance(&left, &right))
}

#[pyfunction]
#[pyo3(signature = (query, references, k=1))]
fn micron_find_top_k(
    query: Vec<u64>,
    references: Vec<Vec<u64>>,
    k: usize,
) -> PyResult<Vec<(usize, u32)>> {
    if query.len() != MICRON_SIGNATURE_WORDS {
        return Err(PyValueError::new_err(
            "Подпись query должна содержать ровно 4 числа u64",
        ));
    }

    let query: [u64; MICRON_SIGNATURE_WORDS] = query.try_into().unwrap();
    let mut matches = Vec::with_capacity(references.len());

    for (index, reference) in references.iter().enumerate() {
        if reference.len() != MICRON_SIGNATURE_WORDS {
            return Err(PyValueError::new_err(format!(
                "Подпись reference[{index}] должна содержать ровно 4 числа u64"
            )));
        }

        let reference: [u64; MICRON_SIGNATURE_WORDS] = reference.clone().try_into().unwrap();
        matches.push((index, micron_hamming_distance(&query, &reference)));
    }

    matches.sort_unstable_by_key(|item| item.1);
    matches.truncate(k);
    Ok(matches)
}

/// Пакетно вычисляет u64-подписи массива векторов.
#[pyfunction]
fn rineto_signatures_batch(vectors: Vec<Vec<f32>>) -> PyResult<Vec<u64>> {
    let mut signatures = Vec::with_capacity(vectors.len());
    for vector in vectors {
        validate_vector(&vector)?;
        signatures.push(vector_signature(&vector));
    }
    Ok(signatures)
}

/// Пакетный top-k по Хэммингу: для каждого query возвращает `(index, hamming)`
/// ближайших эталонных подписей.
#[pyfunction]
#[pyo3(signature = (query_signatures, reference_signatures, k=1))]
fn rineto_hamming_batch(
    query_signatures: Vec<u64>,
    reference_signatures: Vec<u64>,
    k: usize,
) -> Vec<Vec<(usize, u32)>> {
    let mut batch = Vec::with_capacity(query_signatures.len());
    for q_sig in query_signatures {
        let mut matches: Vec<(usize, u32)> = reference_signatures
            .iter()
            .enumerate()
            .map(|(index, r_sig)| (index, (q_sig ^ r_sig).count_ones()))
            .collect();
        matches.sort_unstable_by_key(|item| item.1);
        matches.truncate(k.min(matches.len()));
        batch.push(matches);
    }
    batch
}

#[pymethods]
impl RinetoRouter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Простейший маршрутизатор.
    ///
    /// Позже заменим его маленькой обучаемой моделью или классификатором.
    fn route(&self, text: String) -> RouteDecision {
        let lower = text.to_lowercase();

        if contains_any(&lower, &[
            "код",
            "программа",
            "python",
            "rust",
            "c++",
            "ошибк",
            "компил",
            "функци",
        ]) {
            return RouteDecision {
                domain: DOMAIN_CODE.to_string(),
                expert: "Coder".to_string(),
                template: if contains_any(&lower, &["ошибк", "traceback", "не работает"]) {
                    "FixError".to_string()
                } else {
                    "WriteCode".to_string()
                },
                tools: vec!["compiler".to_string(), "tests".to_string()],
                reason: "Обнаружены признаки задачи программирования".to_string(),
                confidence: 0.82,
            };
        }

        if contains_any(&lower, &[
            "уравн",
            "числ",
            "математ",
            "формул",
            "процент",
            "производн",
            "интеграл",
            "реши",
        ]) {
            return RouteDecision {
                domain: DOMAIN_MATH.to_string(),
                expert: "MathLogic".to_string(),
                template: "SolveAndVerify".to_string(),
                tools: vec!["calculator".to_string(), "z3".to_string()],
                reason: "Обнаружены признаки математической задачи".to_string(),
                confidence: 0.84,
            };
        }

        if contains_any(&lower, &[
            "если",
            "докажи",
            "логик",
            "противореч",
            "услови",
            "план",
            "зависим",
        ]) {
            return RouteDecision {
                domain: DOMAIN_LOGIC.to_string(),
                expert: "Planner".to_string(),
                template: "BuildAndVerifyPlan".to_string(),
                tools: vec!["z3".to_string()],
                reason: "Обнаружены признаки логической или планировочной задачи".to_string(),
                confidence: 0.74,
            };
        }

        if contains_any(&lower, &[
            "напиши рассказ",
            "стих",
            "статья",
            "перефразируй",
            "объясни",
            "диалог",
        ]) {
            return RouteDecision {
                domain: DOMAIN_CHAT.to_string(),
                expert: "Writer".to_string(),
                template: "GenerateText".to_string(),
                tools: Vec::new(),
                reason: "Обнаружена задача генерации или объяснения текста".to_string(),
                confidence: 0.70,
            };
        }

        RouteDecision {
            domain: DOMAIN_GENERAL.to_string(),
            expert: "General".to_string(),
            template: "DirectAnswer".to_string(),
            tools: Vec::new(),
            reason: "Специализированный домен не определён".to_string(),
            confidence: 0.35,
        }
    }
}

fn contains_any(text: &str, words: &[&str]) -> bool {
    words.iter().any(|word| text.contains(word))
}

const RINETO_ASM_MAX_INSTRUCTIONS: usize = 32;

/// Безопасный декларативный маршрут Ринэто.
///
/// Это не машинный ассемблер и не исполнитель произвольного кода. Он хранит
/// только разрешённые шаги оркестрации между Router, экспертами и проверкой.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoASM {
    instructions: Vec<(String, Vec<String>)>,
}

#[pymethods]
impl RinetoASM {
    #[new]
    fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Разбирает программу вида `OP ARG...`, игнорируя пустые строки и `;`-комментарии.
    fn assemble(&mut self, source: String) -> PyResult<Vec<(String, Vec<String>)>> {
        let mut instructions = Vec::new();

        for (line_number, raw_line) in source.lines().enumerate() {
            let line = raw_line
                .split_once(';')
                .map_or(raw_line, |(code, _)| code)
                .trim();
            if line.is_empty() {
                continue;
            }
            if instructions.len() >= RINETO_ASM_MAX_INSTRUCTIONS {
                return Err(PyValueError::new_err(format!(
                    "RinetoASM допускает не более {} инструкций",
                    RINETO_ASM_MAX_INSTRUCTIONS
                )));
            }

            let mut parts = line.split_whitespace();
            let opcode = parts.next().unwrap_or_default().to_ascii_uppercase();
            if !is_rineto_asm_opcode(&opcode) {
                return Err(PyValueError::new_err(format!(
                    "Неизвестный RinetoASM opcode '{}' в строке {}",
                    opcode,
                    line_number + 1
                )));
            }

            instructions.push((
                opcode,
                parts.map(str::to_string).collect::<Vec<_>>(),
            ));
        }

        validate_rineto_asm(&instructions)?;
        self.instructions = instructions.clone();
        Ok(instructions)
    }

    /// Возвращает собранную программу без возможности её произвольно изменить.
    fn instructions(&self) -> Vec<(String, Vec<String>)> {
        self.instructions.clone()
    }

    fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    fn clear(&mut self) {
        self.instructions.clear();
    }

    #[staticmethod]
    fn validate(program: Vec<(String, Vec<String>)>) -> PyResult<bool> {
        validate_rineto_asm(&program).map(|_| true)
    }
}

fn is_rineto_asm_opcode(opcode: &str) -> bool {
    matches!(
        opcode,
        // Управление маршрутом.
        "CLASSIFY_DOMAIN"
            | "SELECT_EXPERT"
            | "SELECT_TEMPLATE"
            | "CALL_TOOL"
            | "VERIFY"
            | "RETURN"
            // Логические связи (из reto_asm Rel*).
            | "ENTAILS"
            | "CONTRADICTS"
            | "CAUSES"
            | "REQUIRES"
            | "VERIFIES"
            | "PRECEDES"
            // Типы знаний (из reto_asm Know*).
            | "KNOW_FACT"
            | "KNOW_RULE"
            | "KNOW_ACTION"
            | "KNOW_RESPONSE"
            | "KNOW_FORMULA"
            | "KNOW_CONSTRAINT"
            | "KNOW_METHOD"
            | "KNOW_SOLUTION"
            | "KNOW_VERIFICATION"
            | "KNOW_CONTRADICTION"
            | "KNOW_ANSWER"
            | "KNOW_START"
            | "KNOW_QUANTITY"
            // Метаданные (из reto_asm Meta*).
            | "META_STYLE"
            | "META_TOOL"
            | "META_REF"
            | "META_FORMULA"
            | "META_PROOF"
    )
}

fn is_knowledge_opcode(opcode: &str) -> bool {
    matches!(
        opcode,
        "KNOW_FACT"
            | "KNOW_RULE"
            | "KNOW_ACTION"
            | "KNOW_RESPONSE"
            | "KNOW_FORMULA"
            | "KNOW_CONSTRAINT"
            | "KNOW_METHOD"
            | "KNOW_SOLUTION"
            | "KNOW_VERIFICATION"
            | "KNOW_CONTRADICTION"
            | "KNOW_ANSWER"
            | "KNOW_START"
            | "KNOW_QUANTITY"
    )
}

fn is_relation_opcode(opcode: &str) -> bool {
    matches!(
        opcode,
        "ENTAILS" | "CONTRADICTS" | "CAUSES" | "REQUIRES" | "VERIFIES" | "PRECEDES"
    )
}

fn is_meta_opcode(opcode: &str) -> bool {
    matches!(
        opcode,
        "META_STYLE" | "META_TOOL" | "META_REF" | "META_FORMULA" | "META_PROOF"
    )
}

/// Термальный контроль Ринэто: адаптация нагрузки к температуре CPU.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoThermal {
    #[pyo3(get, set)]
    pub t_max: f32,

    #[pyo3(get, set)]
    pub t_ambient: f32,

    #[pyo3(get, set)]
    pub beta: f32,

    #[pyo3(get)]
    pub current_temp: f32,

    #[pyo3(get)]
    pub load_factor: f32,

    #[pyo3(get)]
    pub readings: u64,
}

#[pymethods]
impl RinetoThermal {
    #[new]
    #[pyo3(signature = (t_max=80.0, t_ambient=25.0, beta=0.01))]
    fn new(t_max: f32, t_ambient: f32, beta: f32) -> PyResult<Self> {
        let t_max = validate_temperature(t_max, "t_max")?;
        let t_ambient = validate_temperature(t_ambient, "t_ambient")?;
        if beta <= 0.0 || !beta.is_finite() {
            return Err(PyValueError::new_err("beta должен быть положительным числом"));
        }
        if t_max <= t_ambient {
            return Err(PyValueError::new_err(
                "t_max должен быть больше t_ambient",
            ));
        }

        let mut thermal = Self {
            t_max,
            t_ambient,
            beta,
            current_temp: t_ambient,
            load_factor: 1.0,
            readings: 0,
        };
        thermal.update_load_factor();
        Ok(thermal)
    }

    /// Читает температуру из /sys/class/thermal. При недоступности — fallback.
    fn read_temperature(&mut self) -> PyResult<f32> {
        self.readings += 1;
        match std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
            Ok(content) => {
                let parsed = content.trim().parse::<f32>().unwrap_or(25000.0);
                self.current_temp = (parsed / 1000.0).clamp(-50.0, 200.0);
            }
            Err(_) => {
                self.current_temp = 45.0;
            }
        }
        self.update_load_factor();
        Ok(self.current_temp)
    }

    /// Ручная установка температуры (для тестов и симуляции).
    fn set_temperature(&mut self, temperature: f32) -> PyResult<()> {
        if !temperature.is_finite() {
            return Err(PyValueError::new_err("Температура должна быть конечным числом"));
        }
        self.current_temp = temperature.clamp(-50.0, 200.0);
        self.update_load_factor();
        Ok(())
    }

    /// Адаптивный порог уверенности в зависимости от температуры.
    fn confidence_threshold(&self) -> f32 {
        if self.current_temp < 60.0 {
            0.95
        } else if self.current_temp < 75.0 {
            0.90
        } else {
            0.80
        }
    }

    fn mode(&self) -> String {
        if self.load_factor < 0.1 {
            "Superconducting".to_string()
        } else if self.load_factor < 0.5 {
            "Working".to_string()
        } else {
            "Accelerated".to_string()
        }
    }
}

impl RinetoThermal {
    fn update_load_factor(&mut self) {
        let ratio = (self.current_temp - self.t_ambient) / (self.t_max - self.t_ambient);
        self.load_factor = (1.0 - self.beta * ratio * 100.0).clamp(0.0, 1.0);
    }
}

fn validate_temperature(value: f32, name: &str) -> PyResult<f32> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} должен быть конечным числом"
        )));
    }
    Ok(value)
}

const CONTEXT_DIM: usize = 8;

/// Бесконечный контекст Ринэто: фрактальное сжатие истории в фиксированный вектор.
///
/// SHA-256 история → 32 байта → 8 float32 через tanh. EMA-обновление
/// `C = alpha * C_old + (1 - alpha) * C_new`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoContext {
    #[pyo3(get)]
    pub alpha: f32,

    #[pyo3(get)]
    pub updates: u64,

    state: Vec<f32>,
}

#[pymethods]
impl RinetoContext {
    #[new]
    #[pyo3(signature = (alpha=0.999))]
    fn new(alpha: f32) -> PyResult<Self> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(
                "alpha должен быть в диапазоне от 0.0 до 1.0",
            ));
        }
        Ok(Self {
            alpha,
            updates: 0,
            state: vec![0.0; CONTEXT_DIM],
        })
    }

    /// Обновляет состояние новым фрагментом истории.
    fn update(&mut self, text: String) -> Vec<f32> {
        self.updates += 1;
        let compressed = fractal_compress(&text);
        for (index, value) in compressed.iter().enumerate() {
            self.state[index] =
                self.alpha * self.state[index] + (1.0 - self.alpha) * value;
        }
        self.state.clone()
    }

    fn state(&self) -> Vec<f32> {
        self.state.clone()
    }

    fn reset(&mut self) {
        self.state = vec![0.0; CONTEXT_DIM];
        self.updates = 0;
    }

    #[staticmethod]
    fn compress(text: String) -> Vec<f32> {
        fractal_compress(&text)
    }

    fn __repr__(&self) -> String {
        format!("RinetoContext(updates={}, alpha={})", self.updates, self.alpha)
    }
}

fn fractal_compress(text: &str) -> Vec<f32> {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();

    digest
        .chunks(4)
        .map(|chunk| {
            let raw = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            // SHA-256 байты могут интерпретироваться как NaN/inf: заменяем нулём.
            if raw.is_finite() {
                raw.tanh()
            } else {
                0.0
            }
        })
        .collect()
}

const SPHERE_MAX_EDGES_PER_NODE: usize = 64;
const SPHERE_MIN_GENERATE_STEPS: usize = 2;

#[derive(Clone, Debug)]
struct SphereEdge {
    weight: f32,
    fatigue: f32,
}

/// Сфера токенов с рёбрами: граф переходов на единичной сфере.
///
/// Детерминированная версия старого sphere_graph без внешнего RNG.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoSphereGraph {
    #[pyo3(get)]
    pub n_tokens: usize,

    theta: Vec<f32>,
    phi: Vec<f32>,
    edges: Vec<Vec<(usize, SphereEdge)>>,
}

#[pymethods]
impl RinetoSphereGraph {
    #[new]
    #[pyo3(signature = (n_tokens, seed=42))]
    fn new(n_tokens: usize, seed: u64) -> PyResult<Self> {
        if n_tokens == 0 {
            return Err(PyValueError::new_err("Сфера должна содержать хотя бы один токен"));
        }

        let mut rng = XorShift64::new(seed);
        let mut theta = vec![0.0f32; n_tokens];
        let mut phi = vec![0.0f32; n_tokens];

        if n_tokens > 2 {
            theta[0] = 0.0;
            theta[1] = 0.5;
            theta[2] = std::f32::consts::PI;
            for index in 3..n_tokens {
                theta[index] =
                    rng.next_f32() * (std::f32::consts::PI - 0.2) + 0.1;
                phi[index] = rng.next_f32() * 2.0 * std::f32::consts::PI;
            }
        }

        Ok(Self {
            n_tokens,
            theta,
            phi,
            edges: vec![Vec::new(); n_tokens],
        })
    }

    /// Наблюдение за переходом src → dst с весом 1.0.
    fn observe(&mut self, src: usize, dst: usize) {
        self.observe_weighted(src, dst, 1.0);
    }

    /// Наблюдение с произвольным весом.
    fn observe_weighted(&mut self, src: usize, dst: usize, weight: f32) {
        if src >= self.n_tokens || dst >= self.n_tokens || weight <= 0.0 {
            return;
        }

        let existing = self.edges[src].iter_mut().find(|(dst_id, _)| *dst_id == dst);
        if let Some((_, edge)) = existing {
            edge.weight = (edge.weight + weight).min(1_000_000.0);
        } else if self.edges[src].len() < SPHERE_MAX_EDGES_PER_NODE {
            self.edges[src].push((
                dst,
                SphereEdge {
                    weight,
                    fatigue: 0.0,
                },
            ));
        }

        // Сближение позиций на сфере.
        let lr = 0.01;
        self.theta[dst] += (self.theta[src] - self.theta[dst]) * lr;
        self.phi[dst] += (self.phi[src] - self.phi[dst]) * lr;
    }

    fn set_edge_weight(&mut self, src: usize, dst: usize, weight: f32) {
        if src >= self.n_tokens
            || dst >= self.n_tokens
            || !weight.is_finite()
        {
            return;
        }

        let existing = self.edges[src].iter_mut().find(|(dst_id, _)| *dst_id == dst);
        if let Some((_, edge)) = existing {
            edge.weight = weight;
        } else if weight > 0.01 && self.edges[src].len() < SPHERE_MAX_EDGES_PER_NODE {
            self.edges[src].push((
                dst,
                SphereEdge {
                    weight,
                    fatigue: 0.0,
                },
            ));
        }
    }

    /// Жадная генерация последовательности токенов.
    #[pyo3(signature = (start, max_steps=16))]
    fn generate(&mut self, start: usize, max_steps: usize) -> PyResult<Vec<usize>> {
        if start >= self.n_tokens {
            return Err(PyValueError::new_err(format!(
                "Стартовый токен {start} вне диапазона 0..{}",
                self.n_tokens
            )));
        }

        let mut path = vec![start];
        let mut current = start;
        let max_steps = max_steps.min(1024);

        for step in 0..max_steps {
            if self.edges[current].is_empty() {
                break;
            }

            let mut best_dst = None;
            let mut best_weight = f32::NEG_INFINITY;

            for &(dst, ref edge) in &self.edges[current] {
                if path.contains(&dst) {
                    continue;
                }
                let eff = edge.weight * (1.0 - edge.fatigue);
                if eff > best_weight {
                    best_weight = eff;
                    best_dst = Some(dst);
                }
            }

            // Ранние шаги не должны заканчиваться на EOS (токен 2).
            if step < SPHERE_MIN_GENERATE_STEPS && best_dst == Some(2) {
                let alt = self.edges[current]
                    .iter()
                    .filter(|(dst, _)| *dst != 2 && !path.contains(dst))
                    .max_by(|a, b| {
                        (a.1.weight * (1.0 - a.1.fatigue))
                            .partial_cmp(&(b.1.weight * (1.0 - b.1.fatigue)))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(&(alt_dst, _)) = alt {
                    best_dst = Some(alt_dst);
                }
            }

            let dst = match best_dst {
                Some(dst) => dst,
                None => break,
            };

            if let Some((_, edge)) = self.edges[current].iter_mut().find(|(d, _)| *d == dst) {
                edge.fatigue = (edge.fatigue + 0.5).min(1.0);
            }

            path.push(dst);
            current = dst;

            if dst == 2 {
                break;
            }

            for adjacency in &mut self.edges {
                for (_, edge) in adjacency.iter_mut() {
                    edge.fatigue = (edge.fatigue - 0.05).max(0.0);
                }
            }
        }

        Ok(path)
    }

    fn transition_matrix(&self) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0f32; self.n_tokens]; self.n_tokens];
        for src in 0..self.n_tokens {
            for &(dst, ref edge) in &self.edges[src] {
                matrix[src][dst] = edge.weight;
            }
        }
        matrix
    }

    fn edges_of(&self, node: usize) -> Vec<(usize, f32)> {
        if node >= self.n_tokens {
            return Vec::new();
        }
        self.edges[node]
            .iter()
            .map(|(dst, edge)| (*dst, edge.weight))
            .collect()
    }

    fn stats(&self) -> (usize, usize) {
        let n_edges = self.edges.iter().map(|adj| adj.len()).sum();
        (self.n_tokens, n_edges)
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32
    }
}

fn validate_rineto_asm(instructions: &[(String, Vec<String>)]) -> PyResult<()> {
    if instructions.is_empty() {
        return Err(PyValueError::new_err("Пустая RinetoASM-программа"));
    }

    if instructions[0].0 != "CLASSIFY_DOMAIN" {
        return Err(PyValueError::new_err(
            "Маршрут RinetoASM должен начинаться с CLASSIFY_DOMAIN",
        ));
    }

    let mut saw_expert = false;
    let mut saw_template = false;
    let mut saw_tool = false;
    let mut saw_verify = false;
    let mut saw_return = false;

    for (index, (opcode, args)) in instructions.iter().enumerate() {
        if saw_return {
            return Err(PyValueError::new_err(
                "RETURN должен быть последней инструкцией",
            ));
        }

        let is_knowledge = is_knowledge_opcode(opcode);
        let is_relation = is_relation_opcode(opcode);
        let is_meta = is_meta_opcode(opcode);

        let requires_argument = matches!(
            opcode.as_str(),
            "CLASSIFY_DOMAIN" | "SELECT_EXPERT" | "SELECT_TEMPLATE" | "CALL_TOOL"
        );
        let required_args = if is_relation {
            2
        } else if requires_argument || is_knowledge || is_meta {
            1
        } else {
            0
        };
        if args.len() != required_args {
            return Err(PyValueError::new_err(format!(
                "{} ожидает {} аргумент(ов) (инструкция {}), получено {}",
                opcode,
                required_args,
                index + 1,
                args.len()
            )));
        }

        match opcode.as_str() {
            "CLASSIFY_DOMAIN" => {}
            "SELECT_EXPERT" => {
                if saw_expert {
                    return Err(PyValueError::new_err(
                        "Маршрут не должен выбирать несколько основных экспертов",
                    ));
                }
                saw_expert = true;
            }
            "SELECT_TEMPLATE" => {
                if !saw_expert {
                    return Err(PyValueError::new_err(
                        "SELECT_TEMPLATE требует SELECT_EXPERT",
                    ));
                }
                saw_template = true;
            }
            "CALL_TOOL" => {
                if !saw_template {
                    return Err(PyValueError::new_err(
                        "CALL_TOOL требует SELECT_TEMPLATE",
                    ));
                }
                saw_tool = true;
            }
            "VERIFY" => {
                if !saw_template {
                    return Err(PyValueError::new_err(
                        "VERIFY требует выбранный шаблон",
                    ));
                }
                saw_verify = true;
            }
            "RETURN" => {
                if !saw_verify {
                    return Err(PyValueError::new_err(
                        "RETURN требует VERIFY перед выдачей ответа",
                    ));
                }
                saw_return = true;
            }
            // Знания и связи — декларативные факты, не требуют шаблона.
            _ if is_knowledge || is_relation || is_meta => {}
            _ => unreachable!("opcode проверен до валидации"),
        }
    }

    if !saw_expert || !saw_template || !saw_verify || !saw_return {
        return Err(PyValueError::new_err(
            "Маршрут должен содержать EXPERT, TEMPLATE, VERIFY и RETURN",
        ));
    }

    if saw_tool && !instructions.iter().any(|(opcode, _)| opcode == "CALL_TOOL") {
        return Err(PyValueError::new_err("Некорректный CALL_TOOL"));
    }

    Ok(())
}

const BPE_SPECIAL_TOKENS: [&str; 5] = [
"<pad>",
"<unk>",
"<eos>",
"<bos>",
"<sep>",
];
const BPE_MAX_MERGE_LEN: usize = 512;
const BPE_UNK_ID: usize = 1;
// До 200k слияний — достаточно для словаря уровня ~137k токенов
// (как у Qwen ~151k или GPT-4 ~100k).
const BPE_MAX_MERGES_PER_TRAIN: usize = 200_000;

/// Таблицы bytes→unicode (GPT-2 стиль): покрывают все 256 байтов UTF-8.
/// Требуют вызова один раз через OnceLock для скорости.
use std::sync::OnceLock;
static BYTE_TO_CHAR: OnceLock<Vec<char>> = OnceLock::new();
static CHAR_TO_BYTE: OnceLock<HashMap<char, u8>> = OnceLock::new();

fn build_bytes_to_unicode() -> (Vec<char>, HashMap<char, u8>) {
    // GPT-2 bytes-to-unicode: каждый байт 0..255 -> уникальный символ Unicode.
    let mut bs: Vec<u8> = Vec::new();
    for b in 0x21..=0x7e {
        bs.push(b);
    }
    for b in 0xa1..=0xac {
        bs.push(b);
    }
    for b in 0xae..=0xff {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0..256u32 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(0x100 + n);
            n += 1;
        }
    }
    let mut byte_to_char = vec!['\u{fffd}'; 256];
    let mut char_to_byte = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).unwrap_or('\u{fffd}');
        byte_to_char[*b as usize] = ch;
        char_to_byte.insert(ch, *b);
    }
    (byte_to_char, char_to_byte)
}

fn byte_to_char(b: u8) -> char {
    let table = BYTE_TO_CHAR.get_or_init(|| build_bytes_to_unicode().0);
    table[b as usize]
}

fn char_to_byte(ch: char) -> Option<u8> {
    let table = CHAR_TO_BYTE.get_or_init(|| build_bytes_to_unicode().1);
    table.get(&ch).copied()
}

/// Кодирует текст в последовательность mapped-символов байтов UTF-8.
fn text_to_mapped(text: &str) -> Vec<char> {
    text.as_bytes().iter().map(|&b| byte_to_char(b)).collect()
}

/// Декодирует mapped-символы обратно в UTF-8 строку.
fn mapped_to_text(chars: &[char]) -> String {
    let mut bytes = Vec::with_capacity(chars.len());
    for ch in chars {
        match char_to_byte(*ch) {
            Some(b) => bytes.push(b),
            None => continue,
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Семантический хеш Ринэто (SimHash по n-граммам).
///
/// В отличие от SHA-256 (аваланч: 1 символ меняет всё), SimHash сохраняет
/// близость: похожие тексты дают похожие битовые подписи. Это «ринето-замена»
/// SHA-256 для семантической маршрутизации.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoHash;

#[pymethods]
impl RinetoHash {
    #[new]
    fn new() -> Self {
        Self
    }

    /// SimHash-подпись текста (64-бит) по n-граммам (2 и 3 граммы).
    #[staticmethod]
    #[pyo3(signature = (text, gram1=true, gram2=true, gram3=true))]
    fn simhash(text: String, gram1: bool, gram2: bool, gram3: bool) -> u64 {
        simhash_bytes(text.as_bytes(), gram1, gram2, gram3)
    }

    /// SimHash-вектор для маршрутизации: знаки битов -> float-значения.
    /// Похожие тексты -> близкие векторы (cosine ~ близость подписей).
    #[staticmethod]
    #[pyo3(signature = (text, bits=32, gram1=true, gram2=true, gram3=true))]
    fn simhash_vector(text: String, bits: usize, gram1: bool, gram2: bool, gram3: bool) -> PyResult<Vec<f32>> {
        if bits == 0 || bits > 256 {
            return Err(PyValueError::new_err("bits должен быть в (0, 256]"));
        }
        let sig = simhash_bytes(text.as_bytes(), gram1, gram2, gram3);
        let mut vec = Vec::with_capacity(bits);
        for i in 0..bits {
            let bit = (sig >> (i % 64)) & 1;
            vec.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        Ok(vec)
    }

    /// Хэммингово расстояние между двумя подписями.
    #[staticmethod]
    fn hamming(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// Собственный хеш Ринето (64-бит): XOR-свёртка байтов с вращением,
    /// умножением и xorshift-перемешиванием. Формула из белой книги:
    /// каждый байт складывается (XOR) со смещением, затем состояние
    /// перемешивается вращением и умножением на нечётную константу.
    #[staticmethod]
    fn rineto_hash(text: String) -> u64 {
        rineto_hash_bytes(text.as_bytes())
    }

    /// Вектор из ринето-хеша (знаки битов) для сферической маршрутизации.
    /// Похожие тексты НЕ обязаны давать близкие векторы (это полноценный
    /// хеш с лавинным эффектом) — для точных совпадений.
    #[staticmethod]
    #[pyo3(signature = (text, bits=8))]
    fn rineto_hash_vector(text: String, bits: usize) -> PyResult<Vec<f32>> {
        if bits == 0 || bits > 256 {
            return Err(PyValueError::new_err("bits должен быть в (0, 256]"));
        }
        let h = rineto_hash_bytes(text.as_bytes());
        let mut vec = Vec::with_capacity(bits);
        for i in 0..bits {
            let bit = (h >> (i % 64)) & 1;
            vec.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        Ok(vec)
    }

    /// SimHash на формуле Ринето: n-граммы хешируются ринето-формулой
    /// (XOR+вращение+умножение) вместо FNV-1a. Сохраняет семантику n-грамм,
    /// но с битовой «алхимией» Ринето.
    #[staticmethod]
    #[pyo3(signature = (text, gram1=true, gram2=true, gram3=true))]
    fn simhash_rineto(text: String, gram1: bool, gram2: bool, gram3: bool) -> u64 {
        simhash_with_hasher(text.as_bytes(), gram1, gram2, gram3, rineto_hash_bytes)
    }

    /// Вектор SimHash-ринето для сферической маршрутизации (семантический).
    #[staticmethod]
    #[pyo3(signature = (text, bits=32, gram1=true, gram2=true, gram3=true))]
    fn simhash_rineto_vector(text: String, bits: usize, gram1: bool, gram2: bool, gram3: bool) -> PyResult<Vec<f32>> {
        if bits == 0 || bits > 256 {
            return Err(PyValueError::new_err("bits должен быть в (0, 256]"));
        }
        let sig = simhash_with_hasher(text.as_bytes(), gram1, gram2, gram3, rineto_hash_bytes);
        let mut vec = Vec::with_capacity(bits);
        for i in 0..bits {
            let bit = (sig >> (i % 64)) & 1;
            vec.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        Ok(vec)
    }
}

/// Собственный хеш Ринето: XOR-свёртка байтов с вращением и умножением,
/// затем xorshift-перемешивание. Формула из белой книги Рэто-Ориджин —
/// битовая «алхимия»: лавинный эффект при малом входе.
fn rineto_hash_bytes(data: &[u8]) -> u64 {
    const PRIME1: u64 = 0x9E3779B97F4A7C15; // золотое сечение
    const PRIME2: u64 = 0xC2B2AE3D27D4EB4F;
    let mut state: u64 = PRIME1 ^ (data.len() as u64).wrapping_mul(PRIME2);

    for (index, &byte) in data.iter().enumerate() {
        // Каждый байт «замешивается»: XOR со смещением и вращением.
        let mixed = (byte as u64)
            .wrapping_mul(PRIME2)
            .wrapping_add(index as u64)
            .rotate_left(((index as u64) % 64) as u32);
        state ^= mixed;
        state = state.rotate_left(13).wrapping_mul(PRIME1);
        state ^= state >> 7;
    }

    // Финальное перемешивание (лавинный эффект).
    state ^= state >> 29;
    state = state.wrapping_mul(PRIME2);
    state ^= state >> 32;
    state
}

/// SimHash: для каждой n-граммы (с частотой) хешируем в битовый вектор,
/// суммируем веса, знак -> бит. Похожие тексты -> похожие подписи.
fn simhash_bytes(data: &[u8], gram1: bool, gram2: bool, gram3: bool) -> u64 {
    simhash_with_hasher(data, gram1, gram2, gram3, fnv1a_hash)
}

/// Обобщённый SimHash: хешер n-грамм задаётся параметром (FNV-1a или ринето).
fn simhash_with_hasher(
    data: &[u8],
    gram1: bool,
    gram2: bool,
    gram3: bool,
    hasher: fn(&[u8]) -> u64,
) -> u64 {
    // Веса в целых (i32): порядок итерации не влияет на сумму — полностью
    // детерминировано, знак бита не зависит от f32-ошибок округления.
    let mut weights = [0i32; 64];

    if data.is_empty() {
        return 0;
    }

    // Частоты n-грамм. BTreeMap — детерминированный порядок итерации.
    let mut counts: std::collections::BTreeMap<Vec<u8>, usize> = std::collections::BTreeMap::new();
    if gram1 {
        for i in 0..data.len() {
            *counts.entry(vec![data[i]]).or_insert(0) += 1;
        }
    }
    if gram2 && data.len() >= 2 {
        for i in 0..data.len() - 1 {
            *counts.entry(vec![data[i], data[i + 1]]).or_insert(0) += 1;
        }
    }
    if gram3 && data.len() >= 3 {
        for i in 0..data.len() - 2 {
            *counts
                .entry(vec![data[i], data[i + 1], data[i + 2]])
                .or_insert(0) += 1;
        }
    }

    // TF-веса масштабируем до целых (точное суммирование).
    let total = counts.len().max(1) as i64;
    for (gram, count) in &counts {
        // tf_int = round(count / total * 1000) — целочисленная TF.
        let tf_int = ((*count as i64 * 1000) / total) as i32;
        let h = hasher(gram);
        for i in 0..64 {
            if (h >> i) & 1 == 1 {
                weights[i] += tf_int;
            } else {
                weights[i] -= tf_int;
            }
        }
    }

    let mut signature = 0u64;
    for i in 0..64 {
        if weights[i] > 0 {
            signature |= 1u64 << i;
        }
    }
    signature
}

/// Сферическая матрица Ринэто: эталоны на единичной сфере + шаблоны-центры.
///
/// Паттерн «сфера + шаблоны»: векторы нормализуются на единичную сферу,
/// поиск — cosine-близость, шаблоны — сферические центры кластеров.
/// Устойчиво к масштабу признаков (в отличие от L2).
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoSphereMatrix {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub total_adds: u64,

    templates: Vec<Vec<f32>>,
    items: Vec<(Vec<f32>, String, usize)>,
}

#[pymethods]
impl RinetoSphereMatrix {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err(
                "Сферическая матрица должна иметь непустую размерность",
            ));
        }
        Ok(Self {
            dim,
            total_adds: 0,
            templates: Vec::new(),
            items: Vec::new(),
        })
    }

    /// Добавляет сферический шаблон (центр кластера). Возвращает индекс.
    fn add_template(&mut self, template: Vec<f32>) -> PyResult<usize> {
        if template.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} компонентов, получено {}",
                self.dim,
                template.len()
            )));
        }
        let normalized = normalize_to_sphere(&template)?;
        self.templates.push(normalized);
        Ok(self.templates.len() - 1)
    }

    /// Добавляет эталон: нормализует и приписывает к ближайшему шаблону.
    fn add(&mut self, data: Vec<f32>, label: String) -> PyResult<usize> {
        if data.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} компонентов, получено {}",
                self.dim,
                data.len()
            )));
        }
        let normalized = normalize_to_sphere(&data)?;
        self.total_adds += 1;

        let template_idx = if self.templates.is_empty() {
            self.templates.push(normalized.clone());
            0
        } else {
            self.nearest_template(&normalized)?
        };
        self.items.push((normalized, label, template_idx));
        Ok(template_idx)
    }

    /// Cosine-поиск ближайших эталонов: (index, cosine, label).
    #[pyo3(signature = (query, k=1))]
    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(usize, f32, String)>> {
        if query.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} компонентов, получено {}",
                self.dim,
                query.len()
            )));
        }
        if k == 0 || self.items.is_empty() {
            return Ok(Vec::new());
        }
        let query_norm = normalize_to_sphere(&query)?;

        let mut results: Vec<(usize, f32, String)> = self
            .items
            .iter()
            .enumerate()
            .map(|(index, (item, label, _))| {
                let cosine = dot_simd(&query_norm, item);
                (index, cosine, label.clone())
            })
            .collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(results.len()));
        Ok(results)
    }

    /// Маршрутизация к шаблону: (template_idx, cosine).
    fn route_to_template(&self, query: Vec<f32>) -> PyResult<(usize, f32)> {
        if self.templates.is_empty() {
            return Err(PyValueError::new_err("Нет шаблонов"));
        }
        let query_norm = normalize_to_sphere(&query)?;
        let idx = self.nearest_template(&query_norm)?;
        let cosine = dot_simd(&query_norm, &self.templates[idx]);
        Ok((idx, cosine))
    }

    fn template_count(&self) -> usize {
        self.templates.len()
    }

    fn item_count(&self) -> usize {
        self.items.len()
    }

    fn get_template(&self, index: usize) -> PyResult<Vec<f32>> {
        self.templates
            .get(index)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("Шаблон {index} не найден")))
    }
}

impl RinetoSphereMatrix {
    fn nearest_template(&self, query: &[f32]) -> PyResult<usize> {
        let mut best = 0usize;
        let mut best_cos = f32::NEG_INFINITY;
        for (index, template) in self.templates.iter().enumerate() {
            let cosine = dot_simd(query, template);
            if cosine > best_cos {
                best_cos = cosine;
                best = index;
            }
        }
        Ok(best)
    }
}

/// Нормализация вектора на единичную сферу: v / ||v||.
fn normalize_to_sphere(data: &[f32]) -> PyResult<Vec<f32>> {
    let norm_sq: f32 = data.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt();
    if norm <= f32::EPSILON {
        return Err(PyValueError::new_err(
            "Нельзя нормализовать нулевой вектор на сферу",
        ));
    }
    Ok(data.iter().map(|v| v / norm).collect())
}

/// Процессор данных Ринэто (паттерн GLUE DataProcessor).
///
/// Читает JSONL-датасет (instruction/output + category) и создаёт примеры:
/// train/dev/test разбиение + метки категорий. Для обучения генератора ответов.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoDataProcessor {
    #[pyo3(get)]
    pub categories: Vec<String>,

    examples: Vec<(String, String, String)>, // (instruction, output, category)
}

#[pymethods]
impl RinetoDataProcessor {
    #[new]
    fn new() -> Self {
        Self {
            categories: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Загружает примеры из JSONL (как GLUE _create_examples).
    /// Каждая строка: {"instruction": "...", "output": "...", "category": "..."}.
    #[pyo3(signature = (path, set_type="train"))]
    fn load_jsonl(&mut self, path: String, set_type: &str) -> PyResult<usize> {
        let content = std::fs::read_to_string(&path).map_err(|error| {
            PyValueError::new_err(format!("Не удалось прочитать {path}: {error}"))
        })?;

        let mut count = 0;
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || !line.starts_with('{') {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| PyValueError::new_err(format!("Строка {}: {e}", i + 1)))?;

            let instruction = value["instruction"]
                .as_str()
                .unwrap_or("")
                .replace("Пользователь:", "")
                .trim()
                .to_string();
            let output = value["output"].as_str().unwrap_or("").to_string();
            let category = value["category"].as_str().unwrap_or("general").to_string();

            if instruction.is_empty() || output.is_empty() {
                continue;
            }

            self.examples.push((instruction, output, category.clone()));
            if !self.categories.contains(&category) {
                self.categories.push(category);
            }
            count += 1;
            let _ = set_type;
        }
        Ok(count)
    }

    /// Разбивает примеры на train/dev/test (как GLUE get_train/dev/test).
    /// Возвращает кортежи: (train_count, dev_count, test_count).
    #[pyo3(signature = (train_ratio=0.8, dev_ratio=0.1, seed=42))]
    fn split(
        &mut self,
        train_ratio: f64,
        dev_ratio: f64,
        seed: u64,
    ) -> PyResult<(usize, usize, usize)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 || dev_ratio <= 0.0 || train_ratio + dev_ratio >= 1.0 {
            return Err(PyValueError::new_err(
                "train_ratio + dev_ratio должны быть в (0,1)",
            ));
        }
        let mut rng = XorShift64::new(seed);
        for index in 0..self.examples.len() {
            let j = index + (rng.next_u64() as usize % (self.examples.len() - index));
            self.examples.swap(index, j);
        }
        let n = self.examples.len();
        let train = (n as f64 * train_ratio) as usize;
        let dev = (n as f64 * dev_ratio) as usize;
        Ok((train, dev, n - train - dev))
    }

    /// Возвращает примеры train диапазона (instruction, output, category).
    #[pyo3(signature = (start, end))]
    fn get_examples(&self, start: usize, end: usize) -> PyResult<Vec<(String, String, String)>> {
        if end > self.examples.len() {
            return Err(PyValueError::new_err("Выход за границы примеров"));
        }
        Ok(self.examples[start..end].to_vec())
    }

    fn example_count(&self) -> usize {
        self.examples.len()
    }

    fn get_categories(&self) -> Vec<String> {
        self.categories.clone()
    }

    /// Группирует примеры по категориям (как GLUE labels).
    fn examples_by_category(&self) -> PyResult<Vec<(String, usize)>> {
        let mut counts = std::collections::HashMap::new();
        for (_, _, category) in &self.examples {
            *counts.entry(category.clone()).or_insert(0) += 1;
        }
        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(result)
    }

    /// Строит обучающие пары «инструкция → ответ» для генератора.
    /// Возвращает список (текст_промпта, ответ).
    fn build_training_pairs(&self) -> Vec<(String, String)> {
        self.examples
            .iter()
            .map(|(instruction, output, _)| {
                (
                    format!("Пользователь: {instruction} Ryza:"),
                    output.clone(),
                )
            })
            .collect()
    }
}

/// Динамический BPE-токенизатор Ринэто.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoBPE {
    vocab: HashMap<String, usize>,
    id2token: Vec<String>,
    merge_rules: Vec<(String, String, String)>,
    pair_counts: HashMap<(String, String), usize>,
    merge_map_cache: HashMap<(usize, usize), usize>,
    pair_heap: std::collections::BinaryHeap<(usize, std::cmp::Reverse<(String, String)>)>,
    pair_words: HashMap<(String, String), Vec<usize>>,

    #[pyo3(get)]
    pub base_vocab_size: usize,

    #[pyo3(get)]
    pub num_merges: usize,
}

#[pymethods]
impl RinetoBPE {
    #[new]
    fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id2token = Vec::new();

        for (index, special) in BPE_SPECIAL_TOKENS.iter().enumerate() {
            vocab.insert(special.to_string(), index);
            id2token.push(special.to_string());
        }

        // Byte-level база: все 256 байтов UTF-8 -> mapped символы.
        let mut index = BPE_SPECIAL_TOKENS.len();
        for b in 0..=255_u32 {
            let token = byte_to_char(b as u8).to_string();
            vocab.insert(token.clone(), index);
            id2token.push(token);
            index += 1;
        }

        Self {
            base_vocab_size: index,
            vocab,
            id2token,
            merge_rules: Vec::new(),
            pair_counts: HashMap::new(),
            merge_map_cache: HashMap::new(),
            pair_heap: std::collections::BinaryHeap::new(),
            pair_words: HashMap::new(),
            num_merges: 0,
        }
    }

    /// Обучает BPE на корпусе текстов до числа слияний num_merges.
    #[pyo3(signature = (texts, num_merges=100))]
    fn train(&mut self, texts: Vec<String>, num_merges: usize) -> PyResult<usize> {
        let num_merges = num_merges.min(BPE_MAX_MERGES_PER_TRAIN);

        // Начальное разбиение корпуса на слова-последовательности токенов.
        let mut words: Vec<Vec<String>> = Vec::with_capacity(texts.len());
        for text in &texts {
            // Byte-level: каждый байт UTF-8 -> mapped символ, без lowercase.
            let mapped = text_to_mapped(text);
            let mut chars: Vec<String> = Vec::with_capacity(mapped.len());
            for ch in mapped {
                let token = ch.to_string();
                chars.push(if self.vocab.contains_key(&token) {
                    token
                } else {
                    "<unk>".to_string()
                });
            }
            chars.push("<eos>".to_string());
            // Если есть уже обученные правила (повторный train), применяем их,
            // чтобы продолжить обучение с текущего состояния, а не с нуля.
            let mut word = chars;
            for (a, b, merged) in &self.merge_rules {
                word = merge_pair(&word, a, b, merged);
            }
            words.push(word);
        }

        // Первичный подсчёт пар и построение индекса «пара → слова».
        self.pair_counts.clear();
        self.pair_heap.clear();
        self.pair_words.clear();
        for (word_index, word) in words.iter().enumerate() {
            self.add_word_pairs_indexed(word, word_index);
        }

        let mut merges_done = 0;
        for _ in 0..num_merges {
            // Выбираем пару с максимальным счётом из кучи, пропуская устаревшие
            // (счёт не совпадает с текущим pair_counts).
            let best_pair = loop {
                match self.pair_heap.pop() {
                    Some((count, std::cmp::Reverse(pair))) => {
                        if self.pair_counts.get(&pair) == Some(&count) {
                            break Some((pair, count));
                        }
                        // Устаревшая запись в куче — пропускаем.
                    }
                    None => break None,
                }
            };

            let ((a, b), count) = match best_pair {
                Some(pair) => pair,
                None => break,
            };
            if count < 2 {
                break;
            }

            let merged = format!("{}{}", a, b);
            if merged.len() > BPE_MAX_MERGE_LEN {
                break;
            }

            if !self.vocab.contains_key(&merged) {
                let new_id = self.id2token.len();
                self.vocab.insert(merged.clone(), new_id);
                self.id2token.push(merged.clone());
            }
            self.merge_rules.push((a.clone(), b.clone(), merged.clone()));

            // Обрабатываем только слова, содержащие пару (a,b), по индексу.
            // Собираем список слов в отдельный буфер, чтобы не держать заимствование.
            let affected: Vec<usize> = self
                .pair_words
                .get(&(a.clone(), b.clone()))
                .map(|indices| indices.clone())
                .unwrap_or_default();

            for word_index in affected {
                let word = &mut words[word_index];
                if !contains_pair(word, &a, &b) {
                    continue;
                }
                self.remove_word_pairs_indexed(word, word_index);
                *word = merge_pair(word, &a, &b, &merged);
                self.add_word_pairs_indexed(word, word_index);
            }

            // Убираем из индекса обработанную пару — её больше нет в словах.
            self.pair_words.remove(&(a.clone(), b.clone()));

            merges_done += 1;
            self.num_merges += 1;
            let _ = count;
        }

        self.rebuild_merge_map();
        Ok(merges_done)
    }

    /// Обучает до целевого размера словаря (например, 137_000).
    /// Возвращает фактический размер словаря.
    #[pyo3(signature = (texts, target_vocab_size))]
    fn train_to_vocab(
        &mut self,
        texts: Vec<String>,
        target_vocab_size: usize,
    ) -> PyResult<usize> {
        if target_vocab_size <= self.base_vocab_size {
            return Err(PyValueError::new_err(format!(
                "Целевой словарь {target_vocab_size} должен превышать базовый {}",
                self.base_vocab_size
            )));
        }

        let remaining = target_vocab_size.saturating_sub(self.vocab_size());
        if remaining == 0 {
            return Ok(self.vocab_size());
        }

        // Один вызов train с полным остатком: train сам останавливается,
        // когда слияния исчерпаны (куча пуста или все пары с count < 2).
        self.train(texts, remaining)?;

        Ok(self.vocab_size())
    }

    /// Кодирует текст в список ID токенов.
    fn encode(&self, text: String) -> Vec<usize> {
        let mut tokens: Vec<usize> = text_to_mapped(&text)
            .iter()
            .map(|ch| {
                let token = ch.to_string();
                self.vocab.get(&token).copied().unwrap_or(BPE_UNK_ID)
            })
            .collect();
        tokens.push(self.vocab.get("<eos>").copied().unwrap_or(0));

        if self.merge_map_cache.is_empty() {
            return tokens;
        }

        // Жадное применение всех слияний за ограниченное число проходов:
        // за каждый проход заменяем пары согласно merge_map, пока длина не
        // перестанет меняться или не исчерпан лимит проходов.
        let max_passes = 256;
        for _ in 0..max_passes {
            let mut merged = Vec::with_capacity(tokens.len());
            let mut index = 0;
            let mut changed = false;
            while index < tokens.len() {
                if index + 1 < tokens.len() {
                    if let Some(&replacement) =
                        self.merge_map_cache.get(&(tokens[index], tokens[index + 1]))
                    {
                        merged.push(replacement);
                        index += 2;
                        changed = true;
                        continue;
                    }
                }
                merged.push(tokens[index]);
                index += 1;
            }
            tokens = merged;
            if !changed {
                break;
            }
        }

        tokens
    }

    /// Декодирует список ID в текст.
    fn decode(&self, ids: Vec<usize>) -> String {
        let mut chars: Vec<char> = Vec::new();

        for id in ids {
            let token = match self.id2token.get(id) {
                Some(token) => token,
                None => continue,
            };

            if let Some(before_eos) = token.split_once("<eos>") {
                chars.extend(before_eos.0.chars());
                break;
            }

            match token.as_str() {
                "<pad>" | "<unk>" => {}
                "<eos>" | "<bos>" | "<sep>" => break,
                _ => chars.extend(token.chars()),
            }
        }

        mapped_to_text(&chars)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_merge_rules(&self) -> Vec<(String, String, String)> {
        self.merge_rules.clone()
    }

    fn get_stats(&self) -> (usize, usize, usize) {
        (self.vocab.len(), self.num_merges, self.base_vocab_size)
    }

    /// Сохраняет словарь и правила в JSON-файл для повторного использования.
    fn save(&self, path: String) -> PyResult<()> {
        let payload = serde_json::json!({
            "base_vocab_size": self.base_vocab_size,
            "num_merges": self.num_merges,
            "vocab": self.vocab,
            "id2token": self.id2token,
            "merge_rules": self.merge_rules,
        });
        let content = serde_json::to_string_pretty(&payload)
            .map_err(|error| PyValueError::new_err(format!("Сериализация: {error}")))?;
        std::fs::write(&path, content).map_err(|error| {
            PyValueError::new_err(format!("Не удалось записать {path}: {error}"))
        })?;
        Ok(())
    }

    /// Загружает словарь из JSON-файла, сохранённого `save`.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let content = std::fs::read_to_string(&path).map_err(|error| {
            PyValueError::new_err(format!("Не удалось прочитать {path}: {error}"))
        })?;
        let payload: serde_json::Value = serde_json::from_str(&content)
            .map_err(|error| PyValueError::new_err(format!("JSON: {error}")))?;

        let vocab: HashMap<String, usize> = serde_json::from_value(payload["vocab"].clone())
            .map_err(|error| PyValueError::new_err(format!("vocab: {error}")))?;
        let id2token: Vec<String> = serde_json::from_value(payload["id2token"].clone())
            .map_err(|error| PyValueError::new_err(format!("id2token: {error}")))?;
        let merge_rules: Vec<(String, String, String)> =
            serde_json::from_value(payload["merge_rules"].clone())
                .map_err(|error| PyValueError::new_err(format!("merge_rules: {error}")))?;
        let base_vocab_size = payload["base_vocab_size"]
            .as_u64()
            .unwrap_or(vocab.len() as u64) as usize;
        let num_merges = payload["num_merges"].as_u64().unwrap_or(0) as usize;

        if vocab.len() != id2token.len() {
            return Err(PyValueError::new_err(
                "Словарь повреждён: vocab и id2token разной длины",
            ));
        }
        for (token, id) in &vocab {
            if *id >= id2token.len() || id2token[*id] != *token {
                return Err(PyValueError::new_err(format!(
                    "Словарь повреждён: несоответствие токена {token} -> id {id}"
                )));
            }
        }

        let mut bpe = Self {
            vocab,
            id2token,
            merge_rules,
            pair_counts: HashMap::new(),
            merge_map_cache: HashMap::new(),
            pair_heap: std::collections::BinaryHeap::new(),
            pair_words: HashMap::new(),
            base_vocab_size,
            num_merges,
        };
        bpe.rebuild_merge_map();
        Ok(bpe)
    }

    /// Обучает BPE на текстовом файле (по строкам).
    #[pyo3(signature = (path, num_merges=1000))]
    fn train_file(&mut self, path: String, num_merges: usize) -> PyResult<usize> {
        let content = std::fs::read_to_string(&path).map_err(|error| {
            PyValueError::new_err(format!("Не удалось прочитать {path}: {error}"))
        })?;
        let lines: Vec<String> = content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();
        if lines.is_empty() {
            return Err(PyValueError::new_err("Файл не содержит непустых строк"));
        }
        self.train(lines, num_merges)
    }

    /// Обучает BPE с предтокенизацией по словам (как GPT/tiktoken).
    ///
    /// Текст делится на «слова» (с ведущим пробелом, как GPT-2 regex
    /// `\s*\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+`), и BPE работает внутри каждого
    /// слова отдельно. Это даёт реальные словесные токены и большее
    /// разнообразие на том же корпусе.
    #[pyo3(signature = (texts, num_merges=100))]
    fn train_pretokenized(&mut self, texts: Vec<String>, num_merges: usize) -> PyResult<usize> {
        let pretokenized: Vec<String> = texts
            .iter()
            .flat_map(|text| pretokenize_gpt2(text))
            .collect();
        self.train(pretokenized, num_merges)
    }

    /// Делит текст на «слова» в стиле GPT-2 pretokenizer.
    fn pre_tokenize(&self, text: String) -> Vec<String> {
        pretokenize_gpt2(&text)
    }

    /// Пакетное кодирование: возвращает список списков ID.
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<usize>> {
        texts.into_iter().map(|text| self.encode(text)).collect()
    }
}

/// Разбивает текст на «слова» в стиле GPT-2 regex:
/// `\s*\p{L}+ | \p{N}+ | [^\s\p{L}\p{N}]+`.
/// Ведущий пробел привязывается к следующему слову.
fn pretokenize_gpt2(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        let start = index;

        // Ведущие пробелы.
        while index < chars.len() && chars[index].is_whitespace() {
            index += 1;
        }
        let space_end = index;

        if index >= chars.len() {
            if space_end > start {
                words.push(chars[start..space_end].iter().collect());
            }
            break;
        }

        let c = chars[index];
        // Слово (буквы).
        if c.is_alphabetic() {
            while index < chars.len() && chars[index].is_alphabetic() {
                index += 1;
            }
        } else if c.is_numeric() {
            while index < chars.len() && chars[index].is_numeric() {
                index += 1;
            }
        } else {
            // Пунктуация/прочие символы — отдельным «словом».
            while index < chars.len()
                && !chars[index].is_whitespace()
                && !chars[index].is_alphabetic()
                && !chars[index].is_numeric()
            {
                index += 1;
            }
        }

        words.push(chars[start..index].iter().collect());
    }

    words
}

fn merge_pair(
    tokens: &[String],
    a: &str,
    b: &str,
    merged: &str,
) -> Vec<String> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if index + 1 < tokens.len()
            && tokens[index] == a
            && tokens[index + 1] == b
        {
            result.push(merged.to_string());
            index += 2;
        } else {
            result.push(tokens[index].clone());
            index += 1;
        }
    }
    result
}

impl RinetoBPE {
    fn rebuild_merge_map(&mut self) {
        self.merge_map_cache.clear();
        for (a, b, merged) in &self.merge_rules {
            let id_a = self.vocab.get(a).copied();
            let id_b = self.vocab.get(b).copied();
            let id_merged = self.vocab.get(merged).copied();
            if let (Some(a), Some(b), Some(m)) = (id_a, id_b, id_merged) {
                self.merge_map_cache.entry((a, b)).or_insert(m);
            }
        }
    }

    fn add_word_pairs_indexed(&mut self, word: &[String], word_index: usize) {
        for window in word.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            let count = self.pair_counts.entry(pair.clone()).or_insert(0);
            *count += 1;
            let current = *count;
            self.pair_heap
                .push((current, std::cmp::Reverse(pair.clone())));
            self.pair_words
                .entry(pair)
                .or_insert_with(Vec::new)
                .push(word_index);
        }
    }

    fn remove_word_pairs_indexed(&mut self, word: &[String], _word_index: usize) {
        for window in word.windows(2) {
            let key = (window[0].clone(), window[1].clone());
            if let Some(count) = self.pair_counts.get_mut(&key) {
                *count = count.saturating_sub(1);
                let current = *count;
                if current == 0 {
                    self.pair_counts.remove(&key);
                }
            }
        }
    }
}

fn contains_pair(tokens: &[String], a: &str, b: &str) -> bool {
    tokens.windows(2).any(|window| window[0] == a && window[1] == b)
}

/// Эксперт Ринэто: именованная матрица эталонов для одной области знаний.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoExpert {
    #[pyo3(get)]
    pub id: usize,

    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub activation_count: u64,

    #[pyo3(get)]
    pub exemplar_count: u64,

    matrix: RinetoMatrix,
    centroid: Option<Vec<f32>>,
    dim: usize,
}

#[pymethods]
impl RinetoExpert {
    #[new]
    #[pyo3(signature = (id, name, dim))]
    fn new(id: usize, name: String, dim: usize) -> PyResult<Self> {
        Ok(Self {
            id,
            name,
            activation_count: 0,
            exemplar_count: 0,
            matrix: RinetoMatrix::new(dim)?,
            centroid: None,
            dim,
        })
    }

    fn add_exemplar(&mut self, data: Vec<f32>, label: String) -> PyResult<()> {
        if data.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается размерность {}, получено {}",
                self.dim,
                data.len()
            )));
        }
        match &mut self.centroid {
            Some(centroid) => {
                for (c, value) in centroid.iter_mut().zip(&data) {
                    *c = (*c * self.exemplar_count as f32 + value)
                        / (self.exemplar_count + 1) as f32;
                }
            }
            None => {
                self.centroid = Some(data.clone());
            }
        }
        self.exemplar_count += 1;
        self.matrix.add(data, label)
    }

    /// L2-расстояние запроса до центроида эксперта.
    #[pyo3(signature = (query))]
    fn centroid_distance_py(&self, query: Vec<f32>) -> Option<f32> {
        self.centroid_distance(&query)
    }

    /// Обрабатывает запрос и возвращает (hamming, l2, label, confidence).
    fn process(&mut self, query: Vec<f32>) -> PyResult<(u32, f32, String, f32)> {
        self.activation_count += 1;
        let results = self.matrix.forward(query, 1, 4, 2)?;
        match results.into_iter().next() {
            Some((_, hamming, l2, label)) => {
                let confidence = 1.0 / (1.0 + l2);
                Ok((hamming, l2, label, confidence))
            }
            None => Ok((0, 0.0, String::new(), 0.0)),
        }
    }

    fn len(&self) -> usize {
        self.matrix.len()
    }
}

impl RinetoExpert {
    fn centroid_distance(&self, query: &[f32]) -> Option<f32> {
        match &self.centroid {
            Some(centroid) => {
                let mut sum = 0.0f32;
                for (a, b) in query.iter().zip(centroid) {
                    let diff = a - b;
                    sum += diff * diff;
                }
                Some(sum.sqrt())
            }
            None => None,
        }
    }
}

const MOE_DEFAULT_EXPERTS: [&str; 6] = [
    "Смысл",
    "Память",
    "Логика",
    "Характер",
    "Код",
    "Математика",
];

/// Скорректированная уверенность маршрутизации (из RetoMoE):
/// confidence * thermal_factor * specialization_bonus.
/// Бонус специализации для Код (4) и Математика (5) при уверенности > 0.7.
fn adjusted_moe_confidence(confidence: f32, expert_id: usize, thermal_factor: f32) -> f32 {
    let specialization_bonus = match expert_id {
        4 | 5 => {
            if confidence > MOE_SPECIALIZATION_THRESHOLD {
                MOE_SPECIALIZATION_BONUS
            } else {
                1.0
            }
        }
        _ => 1.0,
    };
    confidence * thermal_factor * specialization_bonus
}

/// Маршрутизатор MoE Ринэто: выбирает эксперта по уверенности.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoMoE {
    experts: Vec<RinetoExpert>,

    #[pyo3(get)]
    pub total_routings: u64,

    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub cache_hits: u64,

    #[pyo3(get)]
    pub cache_misses: u64,

    #[pyo3(get)]
    pub superconducting_threshold: f32,

    thermal: RinetoThermal,
    route_cache: Vec<Option<(u64, usize, f32)>>,
}

const MOE_CACHE_SIZE: usize = 256;
const MOE_SPECIALIZATION_BONUS: f32 = 1.15;
const MOE_SPECIALIZATION_THRESHOLD: f32 = 0.7;

#[pymethods]
impl RinetoMoE {
    #[new]
    #[pyo3(signature = (dim, names=Vec::new()))]
    fn new(dim: usize, names: Vec<String>) -> PyResult<Self> {
        let names: Vec<String> = if names.is_empty() {
            MOE_DEFAULT_EXPERTS
                .iter()
                .map(|name| name.to_string())
                .collect()
        } else {
            names
        };
        let mut experts = Vec::with_capacity(names.len());
        for (index, name) in names.into_iter().enumerate() {
            experts.push(RinetoExpert::new(index, name, dim)?);
        }

        Ok(Self {
            experts,
            total_routings: 0,
            dim,
            cache_hits: 0,
            cache_misses: 0,
            superconducting_threshold: 0.5,
            thermal: RinetoThermal::new(80.0, 25.0, 0.01)?,
            route_cache: vec![None; MOE_CACHE_SIZE],
        })
    }

    fn expert_count(&self) -> usize {
        self.experts.len()
    }

    fn add_exemplar(&mut self, expert_id: usize, data: Vec<f32>, label: String) -> PyResult<()> {
        self.expert_mut(expert_id)?.add_exemplar(data, label)?;
        self.invalidate_route_cache();
        Ok(())
    }

    fn set_superconducting_threshold(&mut self, value: f32) {
        self.superconducting_threshold = value.clamp(0.0, 1.0);
    }

    /// Устанавливает температуру для thermal-фактора маршрутизации.
    fn set_thermal_temperature(&mut self, temperature: f32) -> PyResult<()> {
        self.thermal.set_temperature(temperature)
    }

    fn get_thermal_load_factor(&self) -> f32 {
        self.thermal.load_factor
    }

    fn invalidate_route_cache(&mut self) {
        for slot in self.route_cache.iter_mut() {
            *slot = None;
        }
    }

    fn get_route_cache_stats(&self) -> (u64, u64, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }

    /// Маршрутизирует запрос и возвращает (expert_id, name, confidence).
    /// С кэшем по битам, thermal-фактором и specialization bonus (из ryza-reto).
    fn route(&mut self, query: Vec<f32>) -> PyResult<(usize, String, f32)> {
        if query.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается размерность {}, получено {}",
                self.dim,
                query.len()
            )));
        }
        self.total_routings += 1;

        let signature = vector_signature(&query);
        let slot = (signature as usize) % MOE_CACHE_SIZE;

        // Сверхпроводящий режим: повторный запрос из кэша.
        if let Some((cached_sig, cached_expert, cached_conf)) = self.route_cache[slot] {
            if cached_sig == signature {
                self.cache_hits += 1;
                return Ok((cached_expert, self.experts[cached_expert].name.clone(), cached_conf));
            }
        }
        self.cache_misses += 1;

        let thermal_factor = self.thermal.load_factor;
        let mut best_expert = 0usize;
        let mut best_name = String::new();
        let mut best_confidence = f32::NEG_INFINITY;

        for expert in &mut self.experts {
            let (_, _, _, confidence) = expert.process(query.clone())?;
            let adjusted = adjusted_moe_confidence(confidence, expert.id, thermal_factor);
            if adjusted > best_confidence {
                best_confidence = adjusted;
                best_expert = expert.id;
                best_name = expert.name.clone();
            }
        }

        if best_confidence > self.superconducting_threshold {
            self.route_cache[slot] = Some((signature, best_expert, best_confidence));
        }

        Ok((best_expert, best_name, best_confidence))
    }

    /// Маршрутизация по центроидам экспертов (как nearest-centroid классификатор).
    fn route_centroid(&mut self, query: Vec<f32>) -> PyResult<(usize, String, f32)> {
        if query.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается размерность {}, получено {}",
                self.dim,
                query.len()
            )));
        }
        self.total_routings += 1;

        let mut best_expert = 0usize;
        let mut best_name = String::new();
        let mut best_distance = f32::INFINITY;

        for expert in &self.experts {
            if let Some(distance) = expert.centroid_distance(&query) {
                if distance < best_distance {
                    best_distance = distance;
                    best_expert = expert.id;
                    best_name = expert.name.clone();
                }
            }
        }

        let confidence = 1.0 / (1.0 + best_distance);
        Ok((best_expert, best_name, confidence))
    }

    fn expert_stats(&self) -> Vec<(usize, String, u64)> {
        self.experts
            .iter()
            .map(|expert| (expert.id, expert.name.clone(), expert.activation_count))
            .collect()
    }

    fn expert_len(&self, expert_id: usize) -> PyResult<usize> {
        Ok(self.expert_ref(expert_id)?.len())
    }

    /// Возвращает порог уверенности в зависимости от эмоции.
    /// Усталость -> выше порог (осторожнее).
    /// Радость -> ниже порог (дружелюбнее).
    fn emotional_confidence_threshold(&self, emotion: &str) -> f32 {
        let base = self.superconducting_threshold;
        match emotion {
            "усталость" => (base + 0.15).min(1.0),
            "страх" => (base + 0.10).min(1.0),
            "злость" => (base + 0.05).min(1.0),
            "радость" => (base - 0.10).max(0.0),
            "любопытство" => (base - 0.05).max(0.0),
            _ => base,
        }
    }
}

impl RinetoMoE {
    fn expert_mut(&mut self, expert_id: usize) -> PyResult<&mut RinetoExpert> {
        let max_expert = self.experts.len().saturating_sub(1);
        self.experts
            .get_mut(expert_id)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Эксперт {expert_id} не существует (доступны 0..{max_expert})"
                ))
            })
    }

    fn expert_ref(&self, expert_id: usize) -> PyResult<&RinetoExpert> {
        let max_expert = self.experts.len().saturating_sub(1);
        self.experts
            .get(expert_id)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Эксперт {expert_id} не существует (доступны 0..{max_expert})"
                ))
            })
    }
}

/// Шаблонная MoE для чата: 100 экспертов на сфере + шаблоны + самообновление.
///
/// Каждая пара датасета приписывается к эксперту через сферический шаблон.
/// Эксперт «учится по-своему»: хранит только свои пары и может дообучаться
/// на них (self-update). Маршрутизация: query -> SimHash-вектор -> эксперт.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoTemplateMoE {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub num_experts: usize,

    #[pyo3(get)]
    pub total_examples: u64,

    sphere: RinetoSphereMatrix,
    expert_data: Vec<Vec<(String, String)>>,
    exact: HashMap<String, String>,
}

#[pymethods]
impl RinetoTemplateMoE {
    #[new]
    #[pyo3(signature = (dim=32, num_experts=100))]
    fn new(dim: usize, num_experts: usize) -> PyResult<Self> {
        if dim == 0 || num_experts == 0 {
            return Err(PyValueError::new_err(
                "Размерности должны быть больше нуля",
            ));
        }
        Ok(Self {
            dim,
            num_experts,
            total_examples: 0,
            sphere: RinetoSphereMatrix::new(dim)?,
            expert_data: vec![Vec::new(); num_experts],
            exact: HashMap::new(),
        })
    }

    /// Обучение: запрос-ответ приписываются к ближайшему эксперту-шаблону.
    /// Новый эксперт создаётся, если запрос далёк от всех (порог cosine).
    fn learn(&mut self, query: String, answer: String) -> PyResult<usize> {
        self.total_examples += 1;
        self.exact.insert(query.clone(), answer.clone());

        let vector = moe_template_vector(&query, self.dim);

        // Ищем ближайший шаблон (эксперт).
        let mut best_expert = 0usize;
        let mut best_cos = f32::NEG_INFINITY;
        for expert_id in 0..self.sphere.template_count() {
            let template = self.sphere.get_template(expert_id)?;
            let cos = template
                .iter()
                .zip(&vector)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            if cos > best_cos {
                best_cos = cos;
                best_expert = expert_id;
            }
        }

        // Если далёк от всех И есть место — новый эксперт (шаблон).
        let expert_id = if self.sphere.template_count() == 0
            || (best_cos < 0.8 && self.sphere.template_count() < self.num_experts)
        {
            self.sphere.add_template(vector)?;
            self.sphere.template_count() - 1
        } else {
            // Приписываем к ближайшему шаблону.
            self.sphere.add(vector, query.clone())?;
            best_expert
        };

        if expert_id < self.expert_data.len() {
            self.expert_data[expert_id].push((query, answer));
        }
        Ok(expert_id)
    }

    /// Маршрутизация: query -> (expert_id, answer, confidence).
    /// Маршрутизация: query -> (expert_id, answer, confidence).
    fn route(&self, query: String) -> PyResult<(usize, String, f32)> {
        // 1. Точное совпадение.
        if let Some(answer) = self.exact.get(&query) {
            return Ok((0, answer.clone(), 1.0));
        }
        // 1b. Слово-ключ: если слово запроса само является точным запросом.
        let q_lower = query.to_lowercase();
        for (key, answer) in &self.exact {
            if q_lower == key.to_lowercase() {
                return Ok((0, answer.clone(), 0.9));
            }
        }
        // 1c. Синоним: точный запрос, содержащийся в запросе как подстрока.
        for (key, answer) in &self.exact {
            let k = key.to_lowercase();
            if k.len() >= 4 && (q_lower.contains(&k) || k.contains(&q_lower)) {
                return Ok((0, answer.clone(), 0.9));
            }
        }
        // 2. Сфера: ближайшие эталоны -> majority-vote по их ответам.
        let vector = moe_template_vector(&query, self.dim);
        let results = self.sphere.search(vector, 5)?;
        if results.is_empty() {
            return Ok((0, "Извините, не понял.".to_string(), 0.0));
        }
        // Собираем ответы ближайших эталонов и выбираем самый частый.
        let mut answer_counts: std::collections::HashMap<String, (usize, f32)> =
        std::collections::HashMap::new();
        let mut best_conf = 0.0f32;
        for (_, cosine, label) in &results {
            let answer = self
            .exact
            .get(label)
            .cloned()
            .unwrap_or_else(|| label.clone());
            let entry = answer_counts.entry(answer).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += *cosine;
            best_conf = best_conf.max(*cosine);
        }
        let (answer, (count, _)) = answer_counts
        .iter()
        .max_by_key(|(_, (count, _))| *count)
        .map(|(a, c)| (a.clone(), *c))
        .unwrap_or(("Извините, не понял.".to_string(), (0, 0.0)));
        // Эксперт, у которого есть этот ответ.
        let mut expert_id = 0usize;
        for (eid, data) in self.expert_data.iter().enumerate() {
            if data.iter().any(|(_, a)| a == &answer) {
                expert_id = eid;
                break;
            }
        }
        let _ = count;
        Ok((expert_id, answer, best_conf))
    }

    /// Самообновление эксперта: добавляет новый пример к его данным.
    fn self_update(&mut self, expert_id: usize, query: String, answer: String) -> PyResult<()> {        if expert_id >= self.expert_data.len() {
            return Err(PyValueError::new_err(format!(
                "Эксперт {expert_id} не существует (доступны 0..{})",
                self.expert_data.len()
            )));
        }
        self.expert_data[expert_id].push((query.clone(), answer.clone()));
        self.exact.insert(query, answer);
        self.total_examples += 1;
        Ok(())
    }

    fn expert_size(&self, expert_id: usize) -> PyResult<usize> {
        if expert_id >= self.expert_data.len() {
            return Err(PyValueError::new_err("Эксперт не существует"));
        }
        Ok(self.expert_data[expert_id].len())
    }

    fn expert_stats(&self) -> Vec<(usize, usize)> {
        self.expert_data
            .iter()
            .enumerate()
            .map(|(id, data)| (id, data.len()))
            .collect()
    }

    fn num_templates(&self) -> usize {
        self.sphere.template_count()
    }

    fn get_keys(&self) -> Vec<String> {
        self.exact.keys().cloned().collect()
    }

    fn exact_size(&self) -> usize {
        self.exact.len()
    }

    /// Самообучение синонимами: каноничные ответы проецируются на варианты.
    /// Вызывается после обучения всех пар датасета.
    fn learn_synonyms(&mut self) {
        // Находим каноничный ключ по нижнему регистру (кириллица!).
        let find_key = |needle: &str| -> Option<String> {
            let needle_lower = needle.to_lowercase();
            self.exact
                .keys()
                .find(|k| k.to_lowercase() == needle_lower)
                .cloned()
        };

        let mut additions = Vec::new();
        let synonyms = [
            ("привет", vec!["здравствуй", "здравствуйте", "приветик", "приветики", "привет!"]),
            ("как дела", vec!["как поживаешь", "как поживаете", "как жизнь", "как твои дела", "как у тебя дела", "как твои дела?"]),
            ("как дела?", vec!["как твои дела?", "как поживаешь?", "как жизнь?"]),
            ("спасибо", vec!["благодарю", "спасибки", "спасибо!", "спасибо большое"]),
            ("помоги", vec!["помоги мне", "можешь помочь", "нужна помощь", "помогите", "помощь"]),
        ];
        for (canonical, variants) in synonyms {
            if let Some(key) = find_key(canonical) {
                if let Some(answer) = self.exact.get(&key).cloned() {
                    for variant in variants {
                        let v_lower = variant.to_lowercase();
                        if !self
                            .exact
                            .iter()
                            .any(|(k, _)| k.to_lowercase() == v_lower)
                        {
                            additions.push((variant.to_string(), answer.clone()));
                        }
                    }
                }
            }
        }
        for (variant, answer) in additions {
            self.exact.insert(variant, answer);
        }
    }
}

/// SimHash-вектор на сфере для маршрутизации MoE.
fn moe_template_vector(text: &str, dim: usize) -> Vec<f32> {
    let v = rineto_hash_vector_local(text, dim);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v;
    }
    v.iter().map(|x| x / norm).collect()
}

/// Локальная версия SimHash-ринето вектора (без обращения к pyclass).
fn rineto_hash_vector_local(text: &str, dim: usize) -> Vec<f32> {
    let sig = simhash_with_hasher(text.as_bytes(), true, true, true, rineto_hash_bytes);
    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        let bit = (sig >> (i % 64)) & 1;
        vec.push(if bit == 1 { 1.0 } else { -1.0 });
    }
    vec
}

/// Эмоциональный маршрутизатор: объединяет эмоцию и MoE.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoEmotionalRouter {
    emotions: RinetoEmotions,
    moe: RinetoMoE,
}

#[pymethods]
impl RinetoEmotionalRouter {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        Ok(Self {
            emotions: RinetoEmotions::new(),
           moe: RinetoMoE::new(dim, Vec::new())?,
        })
    }

    /// Маршрутизирует с учётом эмоции.
    /// Возвращает (expert_id, expert_name, confidence, emotion).
    #[pyo3(signature = (query, emotion_inputs=None))]
    fn route(
        &mut self,
        query: Vec<f32>,
        emotion_inputs: Option<(f32, f32, f32, f32, f32)>,
    ) -> PyResult<(usize, String, f32, String)> {
        // 1. Вычисляем эмоцию.
        let emotion = match emotion_inputs {
            Some((energy, joy, sadness, tenderness, fear)) => {
                self.emotions.compute(
                    energy, 0.3, 0.5, 0.2, 0.3,
                    joy, sadness, fear, 0.1, tenderness,
                )
            }
            None => self.emotions.current.clone(),
        };

        // 2. Маршрутизируем через MoE (без эмоции — MoE работает по вектору).
        let (expert_id, expert_name, confidence) = self.moe.route(query)?;

        // 3. Возвращаем результат + эмоцию.
        Ok((expert_id, expert_name, confidence, emotion))
    }

    /// Возвращает стиль ответа по текущей эмоции.
    fn response_style(&self) -> (String, String) {
        self.emotions.response_style()
    }
}

/// AdamW-оптимизатор Ринэто с косинусным расписанием, клиппингом и квантовым прыжком.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoOptimizer {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get, set)]
    pub lr: f32,

    #[pyo3(get)]
    pub h_eff: f32,

    #[pyo3(get)]
    pub step_count: usize,

    #[pyo3(get)]
    pub ema_loss: f32,

    params: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    max_v: Vec<f32>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    grad_clip: f32,
    lr_start: f32,
    lr_end: f32,
    warmup_steps: usize,
    total_steps: usize,
    rng_state: u64,
    amsgrad: bool,
    gradient_centralization: bool,
    lookahead_k: usize,
    lookahead_alpha: f32,
    slow_weights: Option<Vec<f32>>,
}

#[pymethods]
impl RinetoOptimizer {
    #[new]
    #[pyo3(signature = (
        dim,
        lr=0.01,
        weight_decay=0.0,
        grad_clip=0.0,
        warmup_steps=0,
        total_steps=100_000,
        seed=42,
        amsgrad=false,
        gradient_centralization=false,
        lookahead_k=0,
        lookahead_alpha=0.5
    ))]
    fn new(
        dim: usize,
        lr: f32,
        weight_decay: f32,
        grad_clip: f32,
        warmup_steps: usize,
        total_steps: usize,
        seed: u64,
        amsgrad: bool,
        gradient_centralization: bool,
        lookahead_k: usize,
        lookahead_alpha: f32,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err(
                "Оптимизатор должен иметь хотя бы один параметр",
            ));
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(PyValueError::new_err("lr должен быть положительным числом"));
        }
        if weight_decay < 0.0 || !weight_decay.is_finite() {
            return Err(PyValueError::new_err(
                "weight_decay не может быть отрицательным",
            ));
        }
        if grad_clip < 0.0 || !grad_clip.is_finite() {
            return Err(PyValueError::new_err(
                "grad_clip не может быть отрицательным",
            ));
        }

        Ok(Self {
            dim,
            lr,
            h_eff: 0.01,
            step_count: 0,
            ema_loss: -1.0,
            params: vec![0.0; dim],
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            max_v: vec![0.0; dim],
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            grad_clip,
            lr_start: lr,
            lr_end: lr * 0.01,
            warmup_steps,
            total_steps,
            rng_state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
            amsgrad,
            gradient_centralization,
            lookahead_k,
            lookahead_alpha,
            slow_weights: if lookahead_k > 0 {
                Some(vec![0.0; dim])
            } else {
                None
            },
        })
    }

    /// Устанавливает начальные параметры.
    fn set_params(&mut self, params: Vec<f32>) -> PyResult<()> {
        if params.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} параметров, получено {}",
                self.dim,
                params.len()
            )));
        }
        if params.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "Параметры должны быть конечными числами",
            ));
        }
        self.params = params;
        Ok(())
    }

    fn get_params(&self) -> Vec<f32> {
        self.params.clone()
    }

    /// Один шаг AdamW по градиенту. Возвращает loss = среднеквадратичную норму градиента.
    fn step(&mut self, gradient: Vec<f32>) -> PyResult<f32> {
        if gradient.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается градиент из {} значений, получено {}",
                self.dim,
                gradient.len()
            )));
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "Градиент должен содержать конечные числа",
            ));
        }

        self.step_count += 1;
        let step = self.step_count;
        self.update_lr();

        // Gradient Centralization (RetoV10): g' = g - mean(g).
        // Стабилизирует обучение, ускоряет сходимость на шумных градиентах.
        let mut centered = gradient;
        if self.gradient_centralization && centered.len() > 1 {
            let mean: f32 = centered.iter().sum::<f32>() / centered.len() as f32;
            for value in &mut centered {
                *value -= mean;
            }
        }

        let mut loss = 0.0f32;
        for index in 0..self.dim {
            let mut grad = centered[index];
            if self.grad_clip > 0.0 {
                grad = grad.clamp(-self.grad_clip, self.grad_clip);
            }
            loss += grad * grad;

            self.m[index] = self.beta1 * self.m[index] + (1.0 - self.beta1) * grad;
            self.v[index] = self.beta2 * self.v[index] + (1.0 - self.beta2) * grad * grad;

            // AMSGrad: v_hat = max(v_hat_prev, v) как в PyTorch Adam(amsgrad=True).
            let mut v_used = self.v[index];
            if self.amsgrad {
                if self.v[index] > self.max_v[index] {
                    self.max_v[index] = self.v[index];
                }
                v_used = self.max_v[index];
            }

            let m_hat = self.m[index] / (1.0 - self.beta1.powi(step as i32));
            let v_hat = v_used / (1.0 - self.beta2.powi(step as i32));

            let update = self.lr * (m_hat / (v_hat.sqrt() + self.epsilon));
            let decay = self.weight_decay * self.lr * self.params[index];
            self.params[index] -= update + decay;
        }

        loss /= self.dim as f32;

        // Lookahead (RetoV10): медленные веса догоняют быстрые каждые k шагов.
        // w_slow <- w_slow + alpha * (w_fast - w_slow).
        if let Some(slow) = &mut self.slow_weights {
            if self.step_count == 1 {
                slow.copy_from_slice(&self.params);
            } else if self.step_count % self.lookahead_k == 0 {
                for index in 0..self.dim {
                    slow[index] += self.lookahead_alpha * (self.params[index] - slow[index]);
                }
                self.params.copy_from_slice(slow);
            }
        }

        if self.ema_loss < 0.0 {
            self.ema_loss = loss;
        } else {
            self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss;
        }

        Ok(loss)
    }

    /// Детерминированный квантовый прыжок: выход из локального минимума.
    fn quantum_leap(&mut self, delta_e: f32) -> PyResult<f32> {
        if !delta_e.is_finite() {
            return Err(PyValueError::new_err("delta_e должен быть конечным числом"));
        }

        let d = self.dim as f32;
        let raw_amplitude = (delta_e / self.h_eff).max(0.0).powf(1.0 / d);
        let ceiling = if delta_e > 1.0 { 0.5 } else { 0.1 };
        let amplitude = raw_amplitude.min(ceiling);

        let mut rng = XorShift64::new(self.rng_state);
        let mut direction = vec![0.0f32; self.dim];
        let mut norm_sq = 0.0f32;
        for value in &mut direction {
            *value = rng.next_f32() * 2.0 - 1.0;
            norm_sq += *value * *value;
        }
        let norm = norm_sq.sqrt() + 1e-8;
        for (index, value) in direction.iter().enumerate() {
            let u = value / norm;
            self.params[index] = self.params[index] * (1.0 - 0.1 * amplitude) + u * amplitude;
        }
        self.rng_state = rng.state;

        Ok(amplitude)
    }

    fn set_h_eff(&mut self, value: f32) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err("h_eff должен быть положительным числом"));
        }
        self.h_eff = value;
        Ok(())
    }

    /// Устанавливает текущую скорость обучения напрямую.
    fn set_learning_rate(&mut self, value: f32) {
        self.lr = value;
    }

    fn debug_state(&self) -> (f32, f32, f32, usize) {
        (self.h_eff, self.ema_loss, self.lr, self.step_count)
    }
}

impl RinetoOptimizer {
    fn update_lr(&mut self) {
        let step = self.step_count;

        if self.warmup_steps > 0 && step <= self.warmup_steps {
            let progress = step as f32 / self.warmup_steps as f32;
            self.lr = self.lr_start + (self.lr - self.lr_start) * progress;
        } else if self.total_steps > self.warmup_steps {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.lr = self.lr_end + cosine * (self.lr_start - self.lr_end);
        }
        self.lr = self.lr.max(self.lr_end);
    }
}

const PROGRAM_CACHE_MAX_ENTRIES: usize = 1024;

/// Кэш скомпилированных RinetoASM-программ по хэшу исходника.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoProgramCache {
    cache: HashMap<u64, Vec<(String, Vec<String>)>>,

    #[pyo3(get)]
    pub cache_hits: u64,

    #[pyo3(get)]
    pub cache_misses: u64,
}

#[pymethods]
impl RinetoProgramCache {
    #[new]
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Возвращает программу из кэша или собирает и сохраняет её.
    fn get_or_assemble(&mut self, source: String) -> PyResult<Vec<(String, Vec<String>)>> {
        let hash = fnv1a_hash(source.as_bytes());
        if let Some(program) = self.cache.get(&hash) {
            self.cache_hits += 1;
            return Ok(program.clone());
        }

        if self.cache.len() >= PROGRAM_CACHE_MAX_ENTRIES {
            self.cache.clear();
        }

        self.cache_misses += 1;
        let mut asm = RinetoASM::new();
        let program = asm.assemble(source)?;
        self.cache.insert(hash, program.clone());
        Ok(program)
    }

    fn cache_size(&self) -> usize {
        self.cache.len()
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Верификатор логических отношений Ринэто.
///
/// Безопасная замена внешнего Z3: проверяет entails/contradicts/requires/verifies
/// через выделение значимых слов и булеву логику. Не выполняет произвольный код.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoZ3Verifier {
    #[pyo3(get)]
    pub checks_performed: u64,

    #[pyo3(get)]
    pub entails_count: u64,

    #[pyo3(get)]
    pub contradicts_count: u64,
}

#[pymethods]
impl RinetoZ3Verifier {
    #[new]
    fn new() -> Self {
        Self {
            checks_performed: 0,
            entails_count: 0,
            contradicts_count: 0,
        }
    }

    /// A entails B: все значимые слова A содержатся в B.
    fn verify_entails(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;
        let a_words = extract_significant_words(&a);
        let b_words = extract_significant_words(&b);

        if a_words.is_empty() {
            self.entails_count += 1;
            return Ok(true);
        }

        let entails = a_words.is_subset(&b_words);
        if entails {
            self.entails_count += 1;
        }
        Ok(entails)
    }

    /// A contradicts B: присутствие отрицания в одном и отсутствие в другом.
    fn verify_contradicts(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;

        let a_has_negation = contains_negation(&a);
        let b_has_negation = contains_negation(&b);
        let contradicts = a_has_negation != b_has_negation;

        if contradicts {
            self.contradicts_count += 1;
        }
        Ok(contradicts)
    }

    /// A causes B: в объединённом тексте есть причинные маркеры.
    fn verify_causes(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;
        let combined = format!("{a} {b}").to_lowercase();
        Ok([
            "потому",
            "поэтому",
            "так как",
            "из-за",
            "следствие",
            "causes",
            "because",
        ]
        .iter()
        .any(|marker| combined.contains(marker)))
    }

    /// A requires B: в объединённом тексте есть маркеры необходимости.
    fn verify_requires(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;
        let combined = format!("{a} {b}").to_lowercase();
        Ok([
            "требует",
            "нужно",
            "необходимо",
            "должен",
            "requires",
            "needs",
            "must",
        ]
        .iter()
        .any(|marker| combined.contains(marker)))
    }

    /// A precedes B: в объединённом тексте есть маркеры порядка.
    fn verify_precedes(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;
        let combined = format!("{a} {b}").to_lowercase();
        Ok([
            "сначала",
            "потом",
            "затем",
            "после",
            "перед",
            "прежде",
        ]
        .iter()
        .any(|marker| combined.contains(marker)))
    }

    /// A verifies B: в объединённом тексте есть маркеры проверки.
    fn verify_verifies(&mut self, a: String, b: String) -> PyResult<bool> {
        self.checks_performed += 1;
        let combined = format!("{a} {b}").to_lowercase();
        Ok([
            "провер",
            "вериф",
            "verif",
            "test",
            "valid",
        ]
        .iter()
        .any(|marker| combined.contains(marker)))
    }

    /// Проверяет текст на наличие цифр (признак QUANTITY).
    fn has_quantity(&mut self, text: String) -> PyResult<bool> {
        self.checks_performed += 1;
        Ok(text.chars().any(|ch| ch.is_ascii_digit()))
    }
}

fn extract_significant_words(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .filter(|word| word.len() > 2)
        .map(|word| word.to_lowercase())
        .collect()
}

fn contains_negation(text: &str) -> bool {
    let lower = text.to_lowercase();

    [
        "не",
        "нет",
        "невозм",
        "ложно",
        "false",
        "never",
        "неверн",
        "ошибк",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

const CPU_CACHE_TILE_SIZE: usize = 128;

/// Кэш-слой поиска Ринэто с учётом попаданий в кэш-линии.
///
/// Безопасная версия старого RetoCPUCache: тайловый бинарный поиск без
/// низкоуровневого `_mm_prefetch`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoCPUCache {
    data: Vec<f32>,
    signatures: Vec<u64>,

    #[pyo3(get)]
    pub n_boundaries: usize,

    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub hits: u64,

    #[pyo3(get)]
    pub misses: u64,

    #[pyo3(get)]
    pub total_ops: u64,
}

#[pymethods]
impl RinetoCPUCache {
    #[new]
    fn new(n_boundaries: usize, dim: usize) -> PyResult<Self> {
        if n_boundaries == 0 || dim == 0 {
            return Err(PyValueError::new_err(
                "Кэш должен иметь непустые размеры",
            ));
        }

        Ok(Self {
            data: vec![0.0; n_boundaries * dim],
            signatures: vec![0; n_boundaries],
            n_boundaries,
            dim,
            hits: 0,
            misses: 0,
            total_ops: 0,
        })
    }

    /// Загружает данные границ и пересчитывает подписи.
    fn load_boundaries(&mut self, data: Vec<f32>) -> PyResult<()> {
        if data.len() != self.n_boundaries * self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} элементов, получено {}",
                self.n_boundaries * self.dim,
                data.len()
            )));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "Данные границ должны быть конечными числами",
            ));
        }

        self.data = data;
        for index in 0..self.n_boundaries {
            let start = index * self.dim;
            self.signatures[index] =
                vector_signature(&self.data[start..start + self.dim]);
        }
        Ok(())
    }

    /// Тайловый двухфазный поиск: бинарный фильтр + точное уточнение.
    fn search(&mut self, x: Vec<f32>) -> PyResult<(usize, f32)> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} элементов, получено {}",
                self.dim,
                x.len()
            )));
        }

        self.total_ops += 1;
        let x_bits = vector_signature(&x);

        let mut best_idx = 0usize;
        let mut best_matches = 0u32;

        // Фаза 1: бинарный фильтр по тайлам.
        for tile_start in (0..self.n_boundaries).step_by(CPU_CACHE_TILE_SIZE) {
            let tile_end = (tile_start + CPU_CACHE_TILE_SIZE).min(self.n_boundaries);
            for index in tile_start..tile_end {
                let matches = 64u32.saturating_sub((x_bits ^ self.signatures[index]).count_ones());
                if matches > best_matches {
                    best_matches = matches;
                    best_idx = index;
                }
            }
        }

        // Фаза 2: точное уточнение.
        let offset = best_idx * self.dim;
        let score: f32 = self.data[offset..offset + self.dim]
            .iter()
            .zip(&x)
            .map(|(a, b)| a * b)
            .sum();
        let norm: f32 = x.iter().map(|value| value * value).sum::<f32>().sqrt() + 1e-8;
        let confidence = score.abs() / norm;

        if confidence > 0.95 {
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        Ok((best_idx, confidence))
    }

    fn get_boundary(&self, index: usize) -> PyResult<Vec<f32>> {
        if index >= self.n_boundaries {
            return Err(PyValueError::new_err(format!(
                "Индекс {index} вне диапазона 0..{}",
                self.n_boundaries
            )));
        }
        let start = index * self.dim;
        Ok(self.data[start..start + self.dim].to_vec())
    }

    fn get_stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.total_ops)
    }
}

const WEIGHTS_MAX_FILE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Безопасная загрузка бинарных f32-весов из файла.
///
/// Настоящий mmap: ленивая загрузка страниц, madvise-предзагрузка, mincore.
#[pyclass]
pub struct RinetoMMapWeights {
    #[allow(dead_code)]
    mmap: Option<memmap2::Mmap>,

    #[pyo3(get)]
    pub path: String,

    #[pyo3(get)]
    pub size: usize,

    #[pyo3(get)]
    pub num_elements: usize,

    #[pyo3(get)]
    pub is_mmap: bool,
}

#[pymethods]
impl RinetoMMapWeights {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let file = std::fs::File::open(&path).map_err(|error| {
            PyValueError::new_err(format!("Файл весов недоступен: {error}"))
        })?;
        let metadata = file.metadata().map_err(|error| {
            PyValueError::new_err(format!("Не удалось получить метаданные: {error}"))
        })?;
        let size = metadata.len() as usize;
        if size == 0 {
            return Err(PyValueError::new_err("Файл весов пуст"));
        }
        if size % 4 != 0 {
            return Err(PyValueError::new_err(format!(
                "Размер файла {size} не кратен 4 байтам (f32)"
            )));
        }
        if size > WEIGHTS_MAX_FILE_BYTES {
            return Err(PyValueError::new_err(format!(
                "Файл слишком велик: {size} байт (лимит {WEIGHTS_MAX_FILE_BYTES})"
            )));
        }

        // Настоящий mmap: ленивая загрузка страниц (как в ryza-reto).
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|error| {
            PyValueError::new_err(format!("Не удалось отобразить файл в память: {error}"))
        })?;

        let num_elements = size / 4;
        Ok(Self {
            mmap: Some(mmap),
            path,
            size,
            num_elements,
            is_mmap: true,
        })
    }

    /// Ленивое чтение f32 по индексу из mmap.
    fn get(&self, index: usize) -> PyResult<f32> {
        if index >= self.num_elements {
            return Err(PyValueError::new_err(format!(
                "Индекс {index} вне диапазона 0..{}",
                self.num_elements
            )));
        }
        if let Some(mmap) = &self.mmap {
            let offset = index * 4;
            let bytes = [
                mmap[offset],
                mmap[offset + 1],
                mmap[offset + 2],
                mmap[offset + 3],
            ];
            Ok(f32::from_le_bytes(bytes))
        } else {
            Err(PyValueError::new_err("mmap не инициализирован"))
        }
    }

    #[pyo3(signature = (start, length))]
    fn get_slice(&self, start: usize, length: usize) -> PyResult<Vec<f32>> {
        let end = start.saturating_add(length);
        if end > self.num_elements {
            return Err(PyValueError::new_err(format!(
                "Срез {start}..{end} вне диапазона 0..{}",
                self.num_elements
            )));
        }
        let mut result = Vec::with_capacity(length);
        if let Some(mmap) = &self.mmap {
            for i in 0..length {
                let offset = (start + i) * 4;
                let bytes = [
                    mmap[offset],
                    mmap[offset + 1],
                    mmap[offset + 2],
                    mmap[offset + 3],
                ];
                result.push(f32::from_le_bytes(bytes));
            }
        }
        Ok(result)
    }

    /// Предзагрузка диапазона в память через madvise(MADV_WILLNEED).
    fn prefetch_range(&self, start: usize, length: usize) -> PyResult<()> {
        if let Some(mmap) = &self.mmap {
            let start_byte = start * 4;
            let end_byte = ((start + length) * 4).min(mmap.len());
            if start_byte >= end_byte {
                return Ok(());
            }
            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    mmap.as_ptr().add(start_byte) as *mut libc::c_void,
                    end_byte - start_byte,
                    libc::MADV_WILLNEED,
                );
            }
            Ok(())
        } else {
            Err(PyValueError::new_err("mmap не инициализирован"))
        }
    }

    /// Подсчёт резидентных страниц через mincore.
    fn resident_pages(&self) -> PyResult<(usize, usize)> {
        if let Some(mmap) = &self.mmap {
            let page_size = 4096usize;
            let num_pages = (mmap.len() + page_size - 1) / page_size;
            #[cfg(unix)]
            unsafe {
                let mut vec = vec![0u8; num_pages];
                let result = libc::mincore(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    vec.as_mut_ptr(),
                );
                if result != 0 {
                    return Ok((0, num_pages));
                }
                let resident = vec.iter().filter(|&&v| v & 1 != 0).count();
                return Ok((resident, num_pages));
            }
            #[cfg(not(unix))]
            {
                let _ = page_size;
                Ok((0, num_pages))
            }
        } else {
            Err(PyValueError::new_err("mmap не инициализирован"))
        }
    }

    /// Прямой доступ к сырым f32-байтам среза без построчной конвертации.
    /// Позволяет numpy/fastcall потребителям работать без Python-цикла.
    #[pyo3(signature = (start, length))]
    fn read_bytes<'py>(&self, py: Python<'py>, start: usize, length: usize) -> PyResult<Bound<'py, PyBytes>> {
        let end = start.saturating_add(length);
        if end > self.num_elements {
            return Err(PyValueError::new_err(format!(
                "Срез {start}..{end} вне диапазона 0..{}",
                self.num_elements
            )));
        }
        let mut bytes = Vec::with_capacity(length * 4);
        if let Some(mmap) = &self.mmap {
            for i in start..end {
                let offset = i * 4;
                bytes.extend_from_slice(&mmap[offset..offset + 4]);
            }
        }
        Ok(PyBytes::new_bound(py, &bytes))
    }

    fn contains(&self, index: usize) -> bool {
        index < self.num_elements
    }

    /// Квантование f32-весов в u16 (экономия памяти 2x, из документа про 16ГБ).
    /// Симметричное: scale = max|w|, w_q = round(w / scale * 32767).
    #[staticmethod]
    fn quantize_u16(weights: Vec<f32>) -> PyResult<(Vec<u16>, f32)> {
        if weights.is_empty() {
            return Err(PyValueError::new_err("Веса не должны быть пустыми"));
        }
        let max_abs = weights
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        if max_abs == 0.0 {
            return Ok((vec![0u16; weights.len()], 0.0));
        }
        let scale = max_abs;
        let quantized = weights
            .iter()
            .map(|v| {
                let scaled = (v / scale * 32767.0).round();
                scaled.clamp(-32767.0, 32767.0) as i32 as u16 & 0xFFFF
            })
            .collect();
        Ok((quantized, scale))
    }

    /// Деквантование u16 обратно в f32.
    #[staticmethod]
    fn dequantize_u16(quantized: Vec<u16>, scale: f32) -> PyResult<Vec<f32>> {
        if scale == 0.0 {
            return Ok(vec![0.0; quantized.len()]);
        }
        Ok(quantized
            .iter()
            .map(|&q| {
                let signed = q as i16 as f32;
                signed / 32767.0 * scale
            })
            .collect())
    }
}

/// Простой полносвязный слой Ринэто с прямым и обратным проходами на Rust.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoLinear {
    #[pyo3(get)]
    pub in_features: usize,

    #[pyo3(get)]
    pub out_features: usize,

    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[pymethods]
impl RinetoLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, init_scale=0.01, seed=42))]
    fn new(in_features: usize, out_features: usize, init_scale: f32, seed: u64) -> PyResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(PyValueError::new_err(
                "Слой должен иметь непустые размерности",
            ));
        }
        if !init_scale.is_finite() || init_scale < 0.0 {
            return Err(PyValueError::new_err(
                "init_scale должен быть неотрицательным числом",
            ));
        }

        let mut rng = XorShift64::new(if seed == 0 { 42 } else { seed });
        let mut weights = Vec::with_capacity(in_features * out_features);
        for _ in 0..in_features * out_features {
            weights.push((rng.next_f32() * 2.0 - 1.0) * init_scale);
        }
        let mut bias = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            bias.push((rng.next_f32() * 2.0 - 1.0) * init_scale);
        }

        Ok(Self {
            in_features,
            out_features,
            weights,
            bias,
        })
    }

    /// Прямой проход: y = x @ W^T + b.
    fn forward(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} входов, получено {}",
                self.in_features,
                x.len()
            )));
        }
        let mut output = vec![0.0f32; self.out_features];
        for (out_index, out) in output.iter_mut().enumerate() {
            let row_start = out_index * self.in_features;
            *out = self.bias[out_index]
                + dot_simd(&self.weights[row_start..row_start + self.in_features], &x);
        }
        Ok(output)
    }

    /// Пакетный прямой проход: каждая строка x обрабатывается одним вызовом.
    fn forward_batch(&self, xs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let mut outputs = Vec::with_capacity(xs.len());
        for x in xs {
            outputs.push(self.forward(x)?);
        }
        Ok(outputs)
    }

    /// Плоский пакетный проход: flat_inputs длиной `batch * in_features`,
    /// возвращает flat длиной `batch * out_features` одним Rust-циклом.
    fn forward_flat_batch(&self, flat_inputs: Vec<f32>) -> PyResult<Vec<f32>> {
        if flat_inputs.is_empty() {
            return Ok(Vec::new());
        }
        if flat_inputs.len() % self.in_features != 0 {
            return Err(PyValueError::new_err(format!(
                "Длина входов {} не кратна in_features={}",
                flat_inputs.len(),
                self.in_features
            )));
        }

        let batch = flat_inputs.len() / self.in_features;
        let output = vec![0.0f32; batch * self.out_features];

        // Многопоточный проход: каждый сэмпл независим (Rayon).
        // Подходит для больших батчей на многоядерном CPU.
        if batch >= 64 {
            use rayon::prelude::*;
            let weights = &self.weights;
            let bias = &self.bias;
            let in_features = self.in_features;
            let out_features = self.out_features;
            let inputs = &flat_inputs;
            let mut out = output;
            out.par_chunks_mut(out_features).enumerate().for_each(|(sample, row)| {
                let x_offset = sample * in_features;
                for out_index in 0..out_features {
                    let w_row = out_index * in_features;
                    row[out_index] = bias[out_index]
                        + dot_simd(&weights[w_row..w_row + in_features],
                                   &inputs[x_offset..x_offset + in_features]);
                }
            });
            return Ok(out);
        }

        // Малый батч: однопоточный (меньше накладных расходов).
        let mut output = output;
        for sample in 0..batch {
            let x_offset = sample * self.in_features;
            let y_offset = sample * self.out_features;
            let sample_x = &flat_inputs[x_offset..x_offset + self.in_features];
            for out_index in 0..self.out_features {
                let w_row = out_index * self.in_features;
                output[y_offset + out_index] = self.bias[out_index]
                    + dot_simd(&self.weights[w_row..w_row + self.in_features], sample_x);
            }
        }

        Ok(output)
    }

    /// Плоский пакетный проход из сырых little-endian f32-байтов.
    /// Принимает bytes (как numpy-буфер) — без поэлементной конвертации.
    fn forward_flat_bytes(&self, flat_bytes: Vec<u8>) -> PyResult<Vec<f32>> {
        if flat_bytes.is_empty() {
            return Ok(Vec::new());
        }
        if flat_bytes.len() % 4 != 0 {
            return Err(PyValueError::new_err(format!(
                "Длина байтов {} не кратна 4 (f32)",
                flat_bytes.len()
            )));
        }

        let count = flat_bytes.len() / 4;
        if count % self.in_features != 0 {
            return Err(PyValueError::new_err(format!(
                "Количество входов {count} не кратно in_features={}",
                self.in_features
            )));
        }

        let batch = count / self.in_features;
        let mut output = vec![0.0f32; batch * self.out_features];

        for sample in 0..batch {
            let x_offset = sample * self.in_features;
            let y_offset = sample * self.out_features;
            for out_index in 0..self.out_features {
                let mut sum = self.bias[out_index];
                let w_row = out_index * self.in_features;
                for in_index in 0..self.in_features {
                    let byte_index = (x_offset + in_index) * 4;
                    let raw = f32::from_le_bytes([
                        flat_bytes[byte_index],
                        flat_bytes[byte_index + 1],
                        flat_bytes[byte_index + 2],
                        flat_bytes[byte_index + 3],
                    ]);
                    sum += raw * self.weights[w_row + in_index];
                }
                output[y_offset + out_index] = sum;
            }
        }

        Ok(output)
    }

    /// Обратный проход по градиенту выхода: возвращает (grad_weights_flat, grad_bias, grad_x).
    fn backward(&self, x: Vec<f32>, grad_output: Vec<f32>) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if x.len() != self.in_features {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} входов, получено {}",
                self.in_features,
                x.len()
            )));
        }
        if grad_output.len() != self.out_features {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} градиентов, получено {}",
                self.out_features,
                grad_output.len()
            )));
        }

        let mut grad_w = vec![0.0f32; self.in_features * self.out_features];
        let mut grad_x = vec![0.0f32; self.in_features];

        for out_index in 0..self.out_features {
            let upstream = grad_output[out_index];
            for in_index in 0..self.in_features {
                grad_w[out_index * self.in_features + in_index] += upstream * x[in_index];
                grad_x[in_index] += upstream * self.weights[out_index * self.in_features + in_index];
            }
        }

        Ok((grad_w, grad_output.clone(), grad_x))
    }

    fn get_weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    fn get_bias(&self) -> Vec<f32> {
        self.bias.clone()
    }

    fn set_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != self.weights.len() {
            return Err(PyValueError::new_err("Неверная размерность весов"));
        }
        self.weights = weights;
        Ok(())
    }

    fn set_bias(&mut self, bias: Vec<f32>) -> PyResult<()> {
        if bias.len() != self.bias.len() {
            return Err(PyValueError::new_err("Неверная размерность bias"));
        }
        self.bias = bias;
        Ok(())
    }

    /// Применяет сдвиг весов по плоскому градиенту (для обучения).
    fn apply_grad(&mut self, grad_w: Vec<f32>, grad_b: Vec<f32>, learning_rate: f32) -> PyResult<()> {
        if grad_w.len() != self.weights.len() || grad_b.len() != self.bias.len() {
            return Err(PyValueError::new_err("Размерности градиентов не совпадают с весами"));
        }
        if !learning_rate.is_finite() {
            return Err(PyValueError::new_err("learning_rate должен быть конечным числом"));
        }
        for (weight, grad) in self.weights.iter_mut().zip(&grad_w) {
            *weight -= learning_rate * grad;
        }
        for (b, grad) in self.bias.iter_mut().zip(&grad_b) {
            *b -= learning_rate * grad;
        }
        Ok(())
    }

    /// Приближённый декодер: hidden @ W^T (без bias). Для сборки изображений.
    fn backward_approx(&self, hidden: Vec<f32>) -> Vec<f32> {
        if hidden.len() != self.out_features {
            return Vec::new();
        }
        let mut out = vec![0.0f32; self.in_features];
        for j in 0..self.in_features {
            let mut sum = 0.0f32;
            for i in 0..self.out_features {
                sum += hidden[i] * self.weights[i * self.in_features + j];
            }
            out[j] = sum;
        }
        out
    }
}

/// LayerNorm Ринэто: нормализация по последней оси с гаммой и бета.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoLayerNorm {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub epsilon: f32,

    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[pymethods]
impl RinetoLayerNorm {
    #[new]
    #[pyo3(signature = (dim, epsilon=1e-5))]
    fn new(dim: usize, epsilon: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err(
                "LayerNorm должен иметь непустую размерность",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PyValueError::new_err(
                "epsilon должен быть положительным числом",
            ));
        }
        Ok(Self {
            dim,
            epsilon,
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
        })
    }

    fn forward(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} входов, получено {}",
                self.dim,
                x.len()
            )));
        }

        // Сумма и сумма квадратов за один проход.
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for value in &x {
            sum += *value;
            sum_sq += value * value;
        }
        let n = x.len() as f32;
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        let std_inv = 1.0 / (variance + self.epsilon).sqrt();

        Ok(x.iter()
            .enumerate()
            .map(|(index, value)| {
                (value - mean) * std_inv * self.gamma[index] + self.beta[index]
            })
            .collect())
    }

    fn set_gamma(&mut self, gamma: Vec<f32>) -> PyResult<()> {
        if gamma.len() != self.dim {
            return Err(PyValueError::new_err("Неверная размерность gamma"));
        }
        self.gamma = gamma;
        Ok(())
    }

    fn set_beta(&mut self, beta: Vec<f32>) -> PyResult<()> {
        if beta.len() != self.dim {
            return Err(PyValueError::new_err("Неверная размерность beta"));
        }
        self.beta = beta;
        Ok(())
    }

    fn get_gamma(&self) -> Vec<f32> {
        self.gamma.clone()
    }

    fn get_beta(&self) -> Vec<f32> {
        self.beta.clone()
    }

    /// Обратный проход LayerNorm.
    /// Возвращает (grad_x, grad_gamma, grad_beta).
    fn backward(
        &self,
        x: Vec<f32>,
        grad_output: Vec<f32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if x.len() != self.dim || grad_output.len() != self.dim {
            return Err(PyValueError::new_err(
                "Размерности x и grad_output должны совпадать с dim",
            ));
        }

        let n = self.dim as f32;
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for value in &x {
            sum += *value;
            sum_sq += value * value;
        }
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        let std_inv = 1.0 / (variance + self.epsilon).sqrt();

        // xhat = (x - mean) * std_inv
        // dxhat = dy * gamma
        // dgamma = sum(dy * xhat), dbeta = sum(dy)
        // dvar = sum(dxhat * (x - mean)) * (-0.5) * std_inv^3
        // dmean = sum(dxhat) * (-std_inv)
        // dx = dxhat * std_inv + dmean/n + dvar * 2*(x-mean)/n

        let mut grad_gamma = vec![0.0f32; self.dim];
        let mut grad_beta = vec![0.0f32; self.dim];
        let mut sum_dxhat = 0.0f32;
        let mut sum_dxhat_xhat = 0.0f32; // sum(dxhat * (x-mean))
        let mut dxhat = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let xhat = (x[i] - mean) * std_inv;
            dxhat[i] = grad_output[i] * self.gamma[i];
            grad_gamma[i] = grad_output[i] * xhat;
            grad_beta[i] = grad_output[i];
            sum_dxhat += dxhat[i];
            sum_dxhat_xhat += dxhat[i] * (x[i] - mean);
        }

        let dvar = sum_dxhat_xhat * (-0.5) * std_inv.powi(3);
        let dmean = sum_dxhat * (-std_inv);

        let mut grad_x = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            grad_x[i] = dxhat[i] * std_inv
                + dmean / n
                + dvar * 2.0 * (x[i] - mean) / n;
        }

        Ok((grad_x, grad_gamma, grad_beta))
    }
}

impl RinetoLayerNorm {
    fn apply_grads(&mut self, ggamma: &[f32], gbeta: &[f32], lr: f32) -> PyResult<()> {
        for (g, grad) in self.gamma.iter_mut().zip(ggamma) {
            *g -= lr * grad;
        }
        for (b, grad) in self.beta.iter_mut().zip(gbeta) {
            *b -= lr * grad;
        }
        Ok(())
    }
}

/// RMSNorm Ринэто (как LlamaRMSNorm): x * rsqrt(mean(x^2) + eps).
/// Без вычитания среднего — проще и быстрее LayerNorm, стандарт для LLM.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoRMSNorm {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub epsilon: f32,

    weight: Vec<f32>,
}

#[pymethods]
impl RinetoRMSNorm {
    #[new]
    #[pyo3(signature = (dim, epsilon=1e-6))]
    fn new(dim: usize, epsilon: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err(
                "RMSNorm должен иметь непустую размерность",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PyValueError::new_err(
                "epsilon должен быть положительным числом",
            ));
        }
        Ok(Self {
            dim,
            epsilon,
            weight: vec![1.0; dim],
        })
    }

    fn forward(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} входов, получено {}",
                self.dim,
                x.len()
            )));
        }

        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let rsqrt = 1.0 / (mean_sq + self.epsilon).sqrt();

        Ok(x.iter()
            .enumerate()
            .map(|(index, value)| value * rsqrt * self.weight[index])
            .collect())
    }

    fn set_weight(&mut self, weight: Vec<f32>) -> PyResult<()> {
        if weight.len() != self.dim {
            return Err(PyValueError::new_err("Неверная размерность weight"));
        }
        self.weight = weight;
        Ok(())
    }

    fn get_weight(&self) -> Vec<f32> {
        self.weight.clone()
    }

    /// Обратный проход RMSNorm: y = x * rsqrt(mean(x^2)+eps) * w.
    /// Возвращает (grad_x, grad_weight).
    fn backward(&self, x: Vec<f32>, grad_output: Vec<f32>) -> PyResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.dim || grad_output.len() != self.dim {
            return Err(PyValueError::new_err(
                "Размерности x и grad_output должны совпадать с dim",
            ));
        }

        let n = self.dim as f32;
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let rsqrt = 1.0 / (mean_sq + self.epsilon).sqrt();
        let rsqrt3 = rsqrt * rsqrt * rsqrt;

        let mut grad_x = vec![0.0f32; self.dim];
        let mut grad_weight = vec![0.0f32; self.dim];

        // grad_weight[i] = grad_output[i] * xhat[i]
        // grad_x[i] = grad_output[i] * w[i] * rsqrt
        //           - (x[i] * rsqrt^3 / n) * sum_j (grad_output[j] * w[j] * x[j])
        let mut dot = 0.0f32;
        for i in 0..self.dim {
            dot += grad_output[i] * self.weight[i] * x[i];
        }
        for i in 0..self.dim {
            grad_weight[i] = grad_output[i] * (x[i] * rsqrt);
            grad_x[i] = grad_output[i] * self.weight[i] * rsqrt
                - (x[i] * rsqrt3 * dot) / n;
        }

        Ok((grad_x, grad_weight))
    }
}

#[allow(dead_code)]
impl RinetoRMSNorm {
    fn apply_grads(&mut self, gweight: &[f32], lr: f32) -> PyResult<()> {
        for (w, grad) in self.weight.iter_mut().zip(gweight) {
            *w -= lr * grad;
        }
        Ok(())
    }
}

/// Embedding-таблица Ринэто: lookup токена в плотный вектор.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoEmbedding {
    #[pyo3(get)]
    pub vocab_size: usize,

    #[pyo3(get)]
    pub dim: usize,

    weights: Vec<f32>,
}

#[pymethods]
impl RinetoEmbedding {
    #[new]
    #[pyo3(signature = (vocab_size, dim, init_scale=0.02, seed=42))]
    fn new(vocab_size: usize, dim: usize, init_scale: f32, seed: u64) -> PyResult<Self> {
        if vocab_size == 0 || dim == 0 {
            return Err(PyValueError::new_err(
                "Embedding должен иметь непустые размерности",
            ));
        }
        if !init_scale.is_finite() || init_scale < 0.0 {
            return Err(PyValueError::new_err(
                "init_scale должен быть неотрицательным числом",
            ));
        }

        let mut rng = XorShift64::new(if seed == 0 { 42 } else { seed });
        let mut weights = Vec::with_capacity(vocab_size * dim);
        for _ in 0..vocab_size * dim {
            weights.push((rng.next_f32() * 2.0 - 1.0) * init_scale);
        }

        Ok(Self {
            vocab_size,
            dim,
            weights,
        })
    }

    fn embed(&self, token_id: usize) -> PyResult<Vec<f32>> {
        if token_id >= self.vocab_size {
            return Err(PyValueError::new_err(format!(
                "Токен {token_id} вне диапазона 0..{}",
                self.vocab_size
            )));
        }
        let start = token_id * self.dim;
        Ok(self.weights[start..start + self.dim].to_vec())
    }

    fn embed_batch(&self, token_ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        let mut batch = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            batch.push(self.embed(token_id)?);
        }
        Ok(batch)
    }

    fn get_weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<f32>) -> PyResult<()> {
        if weights.len() != self.vocab_size * self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} весов, получено {}",
                self.vocab_size * self.dim,
                weights.len()
            )));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "Веса должны быть конечными числами",
            ));
        }
        self.weights = weights;
        Ok(())
    }
}

/// Self-attention Ринэто: scaled dot-product, multi-head с обучаемыми проекциями.
/// Поддерживает KV-кэш для инкрементальной генерации и GQA.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoAttention {
    #[pyo3(get)]
    pub dim: usize,
    #[pyo3(get)]
    pub num_heads: usize,
    #[pyo3(get)]
    pub num_kv_heads: usize,
    #[pyo3(get)]
    pub head_dim: usize,
    #[pyo3(get)]
    pub causal: bool,
    #[pyo3(get)]
    pub sliding_window: usize,

    // Нативные веса (fallback)
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,

    // ГИБРИД: Опциональные квантованные версии
    pub wq_quant: Option<crate::quantized::RinetoBR4>,
    pub wk_quant: Option<crate::quantized::RinetoBR4>,
    pub wv_quant: Option<crate::quantized::RinetoBR4>,
    pub wo_quant: Option<crate::quantized::RinetoIR4>,

    // Bias'ы (всегда f32)
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,

    // KV-кэш для инкрементальной генерации
    #[pyo3(get)]
    pub kv_cache_enabled: bool,
    kv_cache_k: Vec<f32>,
    kv_cache_v: Vec<f32>,
    kv_cache_len: usize,
}
#[pymethods]
impl RinetoAttention {
    #[new]
    #[pyo3(signature = (dim, num_heads=1, num_kv_heads=0, init_scale=0.02, seed=42, causal=false, sliding_window=0))]
    fn new(dim: usize, num_heads: usize, num_kv_heads: usize, init_scale: f32, seed: u64, causal: bool, sliding_window: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err(
                "Attention должен иметь непустую размерность",
            ));
        }
        if num_heads == 0 {
            return Err(PyValueError::new_err(
                "Attention должен иметь хотя бы одну голову",
            ));
        }
        if dim % num_heads != 0 {
            return Err(PyValueError::new_err(format!(
                "dim={dim} не делится на num_heads={num_heads}"
            )));
        }
        // GQA: num_kv_heads должен делить num_heads
        let num_kv_heads = if num_kv_heads == 0 { num_heads } else { num_kv_heads };
        if num_kv_heads > num_heads {
            return Err(PyValueError::new_err(format!(
                "num_kv_heads={num_kv_heads} не может быть больше num_heads={num_heads}"
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(PyValueError::new_err(format!(
                "num_heads={num_heads} должен делиться на num_kv_heads={num_kv_heads}"
            )));
        }
        if !init_scale.is_finite() || init_scale < 0.0 {
            return Err(PyValueError::new_err(
                "init_scale должен быть неотрицательным числом",
            ));
        }

        let head_dim = dim / num_heads;
        let mut rng = XorShift64::new(if seed == 0 { 42 } else { seed });

        let mut init = |count: usize, scale: f32| -> Vec<f32> {
            (0..count)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect()
        };
        // Xavier-подобная нормализация: / sqrt(dim)
        let scale = init_scale / (dim as f32).sqrt();

        Ok(Self {
            dim,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            sliding_window,
            wq: init(dim * dim, scale),
            wk: init(dim * (head_dim * num_kv_heads), scale),
            wv: init(dim * (head_dim * num_kv_heads), scale),
            wo: init(dim * dim, scale),
            wq_quant: None,
            wk_quant: None,
            wv_quant: None,
            wo_quant: None,
            bq: Vec::new(),
            bk: Vec::new(),
            bv: Vec::new(),
            kv_cache_enabled: false,
            kv_cache_k: Vec::new(),
            kv_cache_v: Vec::new(),
            kv_cache_len: 0,
        })
    }

    /// Прямой проход: xs размером (seq_len, dim), возвращает (seq_len, dim).
    fn forward(&self, xs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }

        let seq_len = xs.len();
        let mut q = vec![0.0f32; seq_len * self.dim];
        let mut k = vec![0.0f32; seq_len * self.dim];
        let mut v = vec![0.0f32; seq_len * self.dim];

        // Проекции Q, K, V.
        for i in 0..seq_len {
            for o in 0..self.dim {
                let mut sum_q = 0.0f32;
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for j in 0..self.dim {
                    let x = xs[i][j];
                    sum_q += x * self.wq[o * self.dim + j];
                    sum_k += x * self.wk[o * self.dim + j];
                    sum_v += x * self.wv[o * self.dim + j];
                }
                q[i * self.dim + o] = sum_q;
                k[i * self.dim + o] = sum_k;
                v[i * self.dim + o] = sum_v;
            }
        }

        // Attention по головам (многопоточно: головы независимы).
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut context = vec![0.0f32; seq_len * self.dim];

        if self.num_heads >= 4 {
            use rayon::prelude::*;
            let q_ref = &q;
            let k_ref = &k;
            let v_ref = &v;
            let causal = self.causal;
            let sliding = self.sliding_window;
            let dim = self.dim;
            let head_dim = self.head_dim;
            context
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(token, row)| {
                    for head in 0..self.num_heads {
                        let offset = head * head_dim;
                        // scores
                        let mut scores = vec![0.0f32; seq_len];
                        let q_slice = &q_ref[token * dim + offset..token * dim + offset + head_dim];
                        for j in 0..seq_len {
                            let k_slice = &k_ref[j * dim + offset..j * dim + offset + head_dim];
                            scores[j] = dot_simd(q_slice, k_slice) * scale;
                        }
                        if causal {
                            for j in (token + 1)..seq_len {
                                scores[j] = f32::NEG_INFINITY;
                            }
                        }
                        if sliding > 0 {
                            let start = token.saturating_sub(sliding);
                            for j in 0..start {
                                scores[j] = f32::NEG_INFINITY;
                            }
                        }
                        let weights = softmax(&scores);
                        for h in 0..head_dim {
                            let mut acc = 0.0f32;
                            for j in 0..seq_len {
                                acc += weights[j] * v_ref[j * dim + offset + h];
                            }
                            row[offset + h] = acc;
                        }
                    }
                });
        } else {
            for head in 0..self.num_heads {
                let offset = head * self.head_dim;
                for i in 0..seq_len {
                    // scores[i][j] = dot(q_i, k_j) / sqrt(head_dim)
                    let mut scores = vec![0.0f32; seq_len];
                    let q_slice = &q[i * self.dim + offset..i * self.dim + offset + self.head_dim];
                    for j in 0..seq_len {
                        let k_slice = &k[j * self.dim + offset..j * self.dim + offset + self.head_dim];
                        scores[j] = dot_simd(q_slice, k_slice) * scale;
                    }
                    // Causal mask: будущие позиции -> -inf (как torch.tril в minGPT).
                    if self.causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    // Sliding window (как Mistral): видим только последние W токенов.
                    if self.sliding_window > 0 {
                        let start = i.saturating_sub(self.sliding_window);
                        for j in 0..start {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let weights = softmax(&scores);
                    for h in 0..self.head_dim {
                        let mut acc = 0.0f32;
                        for j in 0..seq_len {
                            acc += weights[j] * v[j * self.dim + offset + h];
                        }
                        context[i * self.dim + offset + h] = acc;
                    }
                }
            }
        }

        // Выходная проекция.
        let mut output = vec![0.0f32; seq_len * self.dim];
        for i in 0..seq_len {
            for o in 0..self.dim {
                let mut sum = 0.0f32;
                for j in 0..self.dim {
                    sum += context[i * self.dim + j] * self.wo[o * self.dim + j];
                }
                output[i * self.dim + o] = sum;
            }
        }

        Ok(output
            .chunks(self.dim)
            .map(|chunk| chunk.to_vec())
            .collect())
    }

    /// Квантует веса attention в br4/ir4 форматы.
    fn quantize_weights(&mut self) -> PyResult<()> {
        use crate::quantized::{RinetoBR4, RinetoIR4};

        self.wq_quant = Some(RinetoBR4::new(self.wq.clone(), vec![self.dim, self.dim])?);
        self.wk_quant = Some(RinetoBR4::new(self.wk.clone(), vec![self.dim, self.dim])?);
        self.wv_quant = Some(RinetoBR4::new(self.wv.clone(), vec![self.dim, self.dim])?);
        self.wo_quant = Some(RinetoIR4::new(self.wo.clone(), vec![self.dim, self.dim])?);

        Ok(())
    }

    /// Проверяет ошибку квантования.
    fn quantization_error(&self) -> PyResult<(f32, f32, f32, f32)> {
        let err_wq = if let Some(ref q) = self.wq_quant {
            let dq = q.dequantize()?;
            self.wq.iter().zip(dq.iter()).map(|(x, y)| (*x - *y).abs()).fold(0.0f32, f32::max)
        } else { 0.0 };

        let err_wk = if let Some(ref q) = self.wk_quant {
            let dq = q.dequantize()?;
            self.wk.iter().zip(dq.iter()).map(|(x, y)| (*x - *y).abs()).fold(0.0f32, f32::max)
        } else { 0.0 };

        let err_wv = if let Some(ref q) = self.wv_quant {
            let dq = q.dequantize()?;
            self.wv.iter().zip(dq.iter()).map(|(x, y)| (*x - *y).abs()).fold(0.0f32, f32::max)
        } else { 0.0 };

        let err_wo = if let Some(ref q) = self.wo_quant {
            let dq = q.dequantize()?;
            self.wo.iter().zip(dq.iter()).map(|(x, y)| (*x - *y).abs()).fold(0.0f32, f32::max)
        } else { 0.0 };

        Ok((err_wq, err_wk, err_wv, err_wo))
    }

    fn get_q(&self) -> Vec<f32> {
        self.wq.clone()
    }

    /// Обратный проход multi-head attention.
    /// Возвращает (grad_x, grad_wq, grad_wk, grad_wv, grad_wo) в плоском виде.
    fn backward(
        &self,
        xs: Vec<Vec<f32>>,
        grad_output: Vec<Vec<f32>>,
    ) -> PyResult<(
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        if xs.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }
        if grad_output.len() != xs.len() {
            return Err(PyValueError::new_err("grad_output и xs разной длины"));
        }

        let seq_len = xs.len();

        // ── forward (повтор) ──
        let mut q = vec![0.0f32; seq_len * self.dim];
        let mut k = vec![0.0f32; seq_len * self.dim];
        let mut v = vec![0.0f32; seq_len * self.dim];
        for i in 0..seq_len {
            for o in 0..self.dim {
                let mut sq = 0.0f32;
                let mut sk = 0.0f32;
                let mut sv = 0.0f32;
                for j in 0..self.dim {
                    let x = xs[i][j];
                    sq += x * self.wq[o * self.dim + j];
                    sk += x * self.wk[o * self.dim + j];
                    sv += x * self.wv[o * self.dim + j];
                }
                q[i * self.dim + o] = sq;
                k[i * self.dim + o] = sk;
                v[i * self.dim + o] = sv;
            }
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // context[i][dim]
        let mut context = vec![0.0f32; seq_len * self.dim];
        // attention_weights[head][i][j]
        let mut attn_weights: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0f32; seq_len]; seq_len]; self.num_heads];

        for head in 0..self.num_heads {
            let offset = head * self.head_dim;
            for i in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                let q_slice = &q[i * self.dim + offset..i * self.dim + offset + self.head_dim];
                for j in 0..seq_len {
                    let k_slice = &k[j * self.dim + offset..j * self.dim + offset + self.head_dim];
                    scores[j] = dot_simd(q_slice, k_slice) * scale;
                }
                // Causal mask в backward: совпадает с forward (minGPT tril).
                if self.causal {
                    for j in (i + 1)..seq_len {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
                // Sliding window в backward: совпадает с forward.
                if self.sliding_window > 0 {
                    let start = i.saturating_sub(self.sliding_window);
                    for j in 0..start {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
                let weights = softmax(&scores);
                for j in 0..seq_len {
                    attn_weights[head][i][j] = weights[j];
                }
                for h in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        acc += weights[j] * v[j * self.dim + offset + h];
                    }
                    context[i * self.dim + offset + h] = acc;
                }
            }
        }

        // ── backward ──
        // grad_context = grad_output @ Wo^T
        let mut grad_context = vec![0.0f32; seq_len * self.dim];
        let mut grad_wo = vec![0.0f32; self.dim * self.dim];
        for i in 0..seq_len {
            for o in 0..self.dim {
                let g = grad_output[i][o];
                for j in 0..self.dim {
                    grad_wo[o * self.dim + j] += context[i * self.dim + j] * g;
                    grad_context[i * self.dim + j] += self.wo[o * self.dim + j] * g;
                }
            }
        }

        // grad_q, grad_k, grad_v, grad_x
        let mut grad_q = vec![0.0f32; seq_len * self.dim];
        let mut grad_k = vec![0.0f32; seq_len * self.dim];
        let mut grad_v = vec![0.0f32; seq_len * self.dim];
        let mut grad_x = vec![vec![0.0f32; self.dim]; seq_len];
        let mut grad_wq = vec![0.0f32; self.dim * self.dim];
        let mut grad_wk = vec![0.0f32; self.dim * self.dim];
        let mut grad_wv = vec![0.0f32; self.dim * self.dim];

        for head in 0..self.num_heads {
            let offset = head * self.head_dim;
            for i in 0..seq_len {
                // grad_weights[i][j] = sum_h grad_context[i][h] * v[j][h]
                let mut grad_weights = vec![0.0f32; seq_len];
                for j in 0..seq_len {
                    let mut acc = 0.0f32;
                    for h in 0..self.head_dim {
                        acc += grad_context[i * self.dim + offset + h]
                            * v[j * self.dim + offset + h];
                    }
                    grad_weights[j] = acc;
                }

                // grad_scores через производную softmax:
                // ds_i = w_i * (g_i - sum_j w_j * g_j)
                let mut grad_scores = vec![0.0f32; seq_len];
                let mut weighted_sum = 0.0f32;
                for j in 0..seq_len {
                    weighted_sum += attn_weights[head][i][j] * grad_weights[j];
                }
                for j in 0..seq_len {
                    grad_scores[j] =
                        attn_weights[head][i][j] * (grad_weights[j] - weighted_sum);
                    grad_scores[j] *= scale; // scores были умножены на scale
                }

                for j in 0..seq_len {
                    let gs = grad_scores[j];
                    for h in 0..self.head_dim {
                        // dq[i] += ds * k[j]
                        grad_q[i * self.dim + offset + h] +=
                            gs * k[j * self.dim + offset + h];
                        // dk[j] += ds * q[i]
                        grad_k[j * self.dim + offset + h] +=
                            gs * q[i * self.dim + offset + h];
                        // dv[j] += w[i][j] * grad_context[i]
                        grad_v[j * self.dim + offset + h] +=
                            attn_weights[head][i][j] * grad_context[i * self.dim + offset + h];
                    }
                }
            }
        }

        // Прокидываем градиенты через QKV-проекции в grad_x и grad_W.
        for i in 0..seq_len {
            for o in 0..self.dim {
                for j in 0..self.dim {
                    grad_wq[o * self.dim + j] += xs[i][j] * grad_q[i * self.dim + o];
                    grad_wk[o * self.dim + j] += xs[i][j] * grad_k[i * self.dim + o];
                    grad_wv[o * self.dim + j] += xs[i][j] * grad_v[i * self.dim + o];
                    grad_x[i][j] +=
                        self.wq[o * self.dim + j] * grad_q[i * self.dim + o]
                            + self.wk[o * self.dim + j] * grad_k[i * self.dim + o]
                            + self.wv[o * self.dim + j] * grad_v[i * self.dim + o];
                }
            }
        }

        Ok((grad_x, grad_wq, grad_wk, grad_wv, grad_wo))
    }
}

impl RinetoAttention {
    fn apply_grads(
        &mut self,
        gwq: &[f32],
        gwk: &[f32],
        gwv: &[f32],
        gwo: &[f32],
        lr: f32,
    ) -> PyResult<()> {
        for (w, g) in self.wq.iter_mut().zip(gwq) {
            *w -= lr * g;
        }
        for (w, g) in self.wk.iter_mut().zip(gwk) {
            *w -= lr * g;
        }
        for (w, g) in self.wv.iter_mut().zip(gwv) {
            *w -= lr * g;
        }
        for (w, g) in self.wo.iter_mut().zip(gwo) {
            *w -= lr * g;
        }
        Ok(())
    }
}

/// Feed-Forward блок Ринэто: Linear → GELU → Linear + residual.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoFFN {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub hidden_dim: usize,

    #[pyo3(get)]
    pub activation: String,

    #[pyo3(get)]
    pub w1_quant: Option<crate::quantized::RinetoBR4>,
    pub w2_quant: Option<crate::quantized::RinetoIR4>,

    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

#[pymethods]
impl RinetoFFN {
    #[new]
    #[pyo3(signature = (dim, hidden_dim, init_scale=0.02, seed=42, activation="gelu"))]
    fn new(
        dim: usize,
        hidden_dim: usize,
        init_scale: f32,
        seed: u64,
        activation: &str,
    ) -> PyResult<Self> {
        if dim == 0 || hidden_dim == 0 {
            return Err(PyValueError::new_err(
                "FFN должен иметь непустые размерности",
            ));
        }
        if !init_scale.is_finite() || init_scale < 0.0 {
            return Err(PyValueError::new_err(
                "init_scale должен быть неотрицательным числом",
            ));
        }
        if activation != "gelu" && activation != "swiglu" {
            return Err(PyValueError::new_err(format!(
                "activation должен быть gelu/swiglu, получено '{activation}'"
            )));
        }

        let mut rng = XorShift64::new(if seed == 0 { 42 } else { seed });
        let scale = init_scale / (dim as f32).sqrt();

        // SwiGLU (как Phi3 gate_up_proj): один слой на (2 * hidden_dim).
        let w1_size = if activation == "swiglu" {
            dim * 2 * hidden_dim
        } else {
            dim * hidden_dim
        };
        let w1: Vec<f32> = (0..w1_size)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let b1 = vec![0.0; if activation == "swiglu" { 2 * hidden_dim } else { hidden_dim }];
        let w2: Vec<f32> = (0..hidden_dim * dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let b2 = vec![0.0; dim];

        Ok(Self {
            dim,
            hidden_dim,
            activation: activation.to_string(),
           w1,
           b1,
           w2,
           b2,
           w1_quant: None,  // ← ДОБАВИТЬ
           w2_quant: None,  // ← ДОБАВИТЬ
        })
    }

    /// Прямой проход с активацией (gelu/swiglu) и residual-связью.
    fn forward(&self, x: Vec<f32>) -> PyResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} входов, получено {}",
                self.dim,
                x.len()
            )));
        }

        // W1 x + b1 -> hidden (2*hidden для swiglu: gate,up).
        let w1_out = self.hidden_dim * if self.activation == "swiglu" { 2 } else { 1 };
        let mut hidden = vec![0.0f32; w1_out];
        for o in 0..w1_out {
            let mut sum = self.b1[o];
            for j in 0..self.dim {
                sum += x[j] * self.w1[o * self.dim + j];
            }
            hidden[o] = sum;
        }

        // Активация.
        let hidden_dim = self.hidden_dim;
        if self.activation == "swiglu" {
            // gate = gelu(hidden[0..h]), up = hidden[h..2h]; out = gate * up.
            let mut activated = vec![0.0f32; hidden_dim];
            for o in 0..hidden_dim {
                let gate = gelu(hidden[o]);
                let up = hidden[hidden_dim + o];
                activated[o] = gate * up;
            }
            hidden = activated;
        } else {
            for value in &mut hidden {
                *value = gelu(*value);
            }
        }

        // W2 hidden + b2 + residual
        let mut output = vec![0.0f32; self.dim];
        for o in 0..self.dim {
            let mut sum = self.b2[o];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[o * self.hidden_dim + j];
            }
            output[o] = sum + x[o];
        }

        Ok(output)
    }

    fn get_w1(&self) -> Vec<f32> {
        self.w1.clone()
    }

    fn get_w2(&self) -> Vec<f32> {
        self.w2.clone()
    }

    fn set_w1(&mut self, w1: Vec<f32>) -> PyResult<()> {
        let expected_w1_len = if self.activation == "swiglu" {
            self.dim * 2 * self.hidden_dim
        } else {
            self.dim * self.hidden_dim
        };

        if w1.len() != expected_w1_len {
            return Err(PyValueError::new_err(format!(
                "Неверная размерность w1: ожидается {}, получено {}",
                expected_w1_len,
                w1.len()
            )));
        }

        self.w1 = w1;
        Ok(())
    }

    /// Обратный проход FFN (gelu или swiglu) + residual внутри.
    /// Возвращает (grad_w1, grad_b1, grad_w2, grad_b2, grad_x).
    fn backward(
        &self,
        x: Vec<f32>,
        grad_output: Vec<f32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        if x.len() != self.dim || grad_output.len() != self.dim {
            return Err(PyValueError::new_err(
                "Размерности x и grad_output должны совпадать с dim",
            ));
        }

        let hidden_dim = self.hidden_dim;
        let w1_out = hidden_dim * if self.activation == "swiglu" { 2 } else { 1 };
        let is_swiglu = self.activation == "swiglu";

        // Повторяем forward: pre_act (w1_out), затем activated (hidden_dim).
        let mut pre_act = vec![0.0f32; w1_out];
        for o in 0..w1_out {
            let mut sum = self.b1[o];
            for j in 0..self.dim {
                sum += x[j] * self.w1[o * self.dim + j];
            }
            pre_act[o] = sum;
        }

        let mut activated = vec![0.0f32; hidden_dim];
        if is_swiglu {
            // gate = gelu(pre_act[0..h]), up = pre_act[h..2h]; activated = gate * up.
            for o in 0..hidden_dim {
                activated[o] = gelu(pre_act[o]) * pre_act[hidden_dim + o];
            }
        } else {
            for o in 0..hidden_dim {
                activated[o] = gelu(pre_act[o]);
            }
        }

        // grad_w2 = activated^T @ grad_output ; grad_b2 = sum(grad_output)
        let mut grad_w2 = vec![0.0f32; hidden_dim * self.dim];
        let mut grad_b2 = vec![0.0f32; self.dim];
        let mut grad_activated = vec![0.0f32; hidden_dim];
        for o in 0..self.dim {
            let g = grad_output[o];
            grad_b2[o] += g;
            for j in 0..hidden_dim {
                grad_w2[o * hidden_dim + j] += activated[j] * g;
                grad_activated[j] += self.w2[o * hidden_dim + j] * g;
            }
        }

        // d(activated)/d(pre_act).
        let mut grad_pre = vec![0.0f32; w1_out];
        if is_swiglu {
            for o in 0..hidden_dim {
                let gate = pre_act[o];
                let up = pre_act[hidden_dim + o];
                // activated = gelu(gate) * up
                grad_pre[o] = grad_activated[o] * up * gelu_derivative(gate);
                grad_pre[hidden_dim + o] = grad_activated[o] * gelu(gate);
            }
        } else {
            for o in 0..hidden_dim {
                grad_pre[o] = grad_activated[o] * gelu_derivative(pre_act[o]);
            }
        }

        // grad_w1 = pre_act_grad^T @ x ; grad_b1 = sum(pre_act_grad)
        let mut grad_w1 = vec![0.0f32; self.dim * w1_out];
        let mut grad_b1 = vec![0.0f32; w1_out];
        let mut grad_x = vec![0.0f32; self.dim];
        for o in 0..w1_out {
            let g = grad_pre[o];
            grad_b1[o] += g;
            for j in 0..self.dim {
                grad_w1[o * self.dim + j] += x[j] * g;
                grad_x[j] += self.w1[o * self.dim + j] * g;
            }
        }

        // residual: x участвует напрямую.
        for i in 0..self.dim {
            grad_x[i] += grad_output[i];
        }

        Ok((grad_w1, grad_b1, grad_w2, grad_b2, grad_x))
    }
}

impl RinetoFFN {
    fn apply_grads(
        &mut self,
        gw1: &[f32],
        gb1: &[f32],
        gw2: &[f32],
        gb2: &[f32],
        lr: f32,
    ) -> PyResult<()> {
        for (w, g) in self.w1.iter_mut().zip(gw1) {
            *w -= lr * g;
        }
        for (b, g) in self.b1.iter_mut().zip(gb1) {
            *b -= lr * g;
        }
        for (w, g) in self.w2.iter_mut().zip(gw2) {
            *w -= lr * g;
        }
        for (b, g) in self.b2.iter_mut().zip(gb2) {
            *b -= lr * g;
        }
        Ok(())
    }
}

fn gelu_derivative(value: f32) -> f32 {
    // gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    // производная: 0.5*(1+tanh(a)) + 0.5*x*(1-tanh(a)^2)*a', где a' = sqrt(2/pi)*(1+3*0.044715*x^2)
    let c = std::f32::consts::FRAC_2_SQRT_PI;
    let a = c * (value + 0.044715 * value * value * value);
    let tanh_a = a.tanh();
    let a_prime = c * (1.0 + 3.0 * 0.044715 * value * value);
    0.5 * (1.0 + tanh_a) + 0.5 * value * (1.0 - tanh_a * tanh_a) * a_prime
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044715 * value * value * value)).tanh())
}

/// RMSNorm forward (как LlamaRMSNorm): x * rsqrt(mean(x^2)+eps) * w.
fn rmsnorm_forward(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(value, w)| value * rsqrt * w)
        .collect()
}

/// RMSNorm backward: возвращает (grad_x, grad_weight).
fn rmsnorm_backward(x: &[f32], grad_output: &[f32], weight: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();
    let rsqrt3 = rsqrt * rsqrt * rsqrt;

    let mut dot = 0.0f32;
    for i in 0..x.len() {
        dot += grad_output[i] * weight[i] * x[i];
    }

    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_weight = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        grad_weight[i] = grad_output[i] * (x[i] * rsqrt);
        grad_x[i] = grad_output[i] * weight[i] * rsqrt - (x[i] * rsqrt3 * dot) / n;
    }
    (grad_x, grad_weight)
}

/// Полный блок трансформера Ринэто:
/// (RMSNorm|LayerNorm) → Attention → residual → Norm → FFN → residual.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoTransformerBlock {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub use_rmsnorm: bool,

    #[pyo3(get)]
    pub activation: String,

    attention: RinetoAttention,
    ffn: RinetoFFN,
    norm1: RinetoLayerNorm,
    norm2: RinetoLayerNorm,
    rms_weight1: Vec<f32>,
    rms_weight2: Vec<f32>,
    rms_eps: f32,
}

#[pymethods]
impl RinetoTransformerBlock {
    #[new]
    #[pyo3(signature = (dim, num_heads=1, hidden_dim=0, init_scale=0.02, seed=42, causal=false, activation="gelu", use_rmsnorm=false, sliding_window=0))]
    fn new(
        dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        init_scale: f32,
        seed: u64,
        causal: bool,
        activation: &str,
        use_rmsnorm: bool,
        sliding_window: usize,
    ) -> PyResult<Self> {
        let hidden_dim = if hidden_dim == 0 { dim * 4 } else { hidden_dim };
        Ok(Self {
            dim,
            use_rmsnorm,
            activation: activation.to_string(),
            attention: RinetoAttention::new(dim, num_heads, num_heads, init_scale, seed, causal, sliding_window)?,
            ffn: RinetoFFN::new(dim, hidden_dim, init_scale, seed + 1, activation)?,
            norm1: RinetoLayerNorm::new(dim, 1e-5)?,
            norm2: RinetoLayerNorm::new(dim, 1e-5)?,
            rms_weight1: vec![1.0; dim],
            rms_weight2: vec![1.0; dim],
            rms_eps: 1e-6,
        })
    }

    /// Прямой проход для последовательности векторов (seq_len, dim).
    fn forward(&self, xs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }

        // Нормализация перед attention.
        let mut normed = Vec::with_capacity(xs.len());
        for row in &xs {
            if self.use_rmsnorm {
                normed.push(rmsnorm_forward(row, &self.rms_weight1, self.rms_eps));
            } else {
                normed.push(self.norm1.forward(row.clone())?);
            }
        }

        let attended = self.attention.forward(normed)?;

        // Residual 1.
        let mut after_attn = Vec::with_capacity(xs.len());
        for (original, attn) in xs.iter().zip(&attended) {
            let mut combined = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                combined[i] = original[i] + attn[i];
            }
            after_attn.push(combined);
        }

        // Нормализация перед FFN.
        let mut normed2 = Vec::with_capacity(xs.len());
        for row in &after_attn {
            if self.use_rmsnorm {
                normed2.push(rmsnorm_forward(row, &self.rms_weight2, self.rms_eps));
            } else {
                normed2.push(self.norm2.forward(row.clone())?);
            }
        }

        // FFN + residual 2.
        let mut output = Vec::with_capacity(xs.len());
        for (pre_ffn, normed_row) in after_attn.into_iter().zip(normed2) {
            let ffn_out = self.ffn.forward(normed_row)?;
            let mut combined = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                combined[i] = pre_ffn[i] + ffn_out[i];
            }
            output.push(combined);
        }

        Ok(output)
    }

    /// Обратный проход блока. Возвращает (grad_x, градиенты параметров).
    /// Градиенты в плоском списке:
    /// [attention: wq,wk,wv,wo] + [ffn: w1,b1,w2,b2] +
    /// [norm1: gamma,beta] + [norm2: gamma,beta].
    #[pyo3(signature = (xs, grad_output))]
    fn backward(
        &self,
        xs: Vec<Vec<f32>>,
        grad_output: Vec<Vec<f32>>,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        if xs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // ── повтор forward ──
        let mut normed = Vec::with_capacity(xs.len());

        for row in &xs {
            if self.use_rmsnorm {
                normed.push(rmsnorm_forward(
                    row,
                    &self.rms_weight1,
                    self.rms_eps,
                ));
            } else {
                normed.push(self.norm1.forward(row.clone())?);
            }
        }
        let attended = self.attention.forward(normed.clone())?;

        let mut after_attn = Vec::with_capacity(xs.len());
        for (original, attn) in xs.iter().zip(&attended) {
            let mut combined = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                combined[i] = original[i] + attn[i];
            }
            after_attn.push(combined);
        }

        let mut normed2 = Vec::with_capacity(xs.len());

        for row in &after_attn {
            if self.use_rmsnorm {
                normed2.push(rmsnorm_forward(
                    row,
                    &self.rms_weight2,
                    self.rms_eps,
                ));
            } else {
                normed2.push(self.norm2.forward(row.clone())?);
            }
        }

        // ── backward ──
        // Выход: out = after_attn + ffn(normed2); ffn включает residual normed2.
        // grad_after_attn = grad_output; grad_normed2 = grad_output (через residual ffn)
        let mut grad_after_attn = grad_output.clone();
        let mut grad_ffn_params: Vec<f32> = Vec::new();

        // FFN.backward возвращает (gw1,gb1,gw2,gb2,gx); gx уже включает residual.
        let mut grad_normed2_from_ffn = vec![vec![0.0f32; self.dim]; xs.len()];
        for (index, (normed_row, g)) in normed2.iter().zip(&grad_output).enumerate() {
            let (gw1, gb1, gw2, gb2, gx) = self.ffn.backward(normed_row.clone(), g.clone())?;
            if index == 0 {
                grad_ffn_params.extend(gw1);
                grad_ffn_params.extend(gb1);
                grad_ffn_params.extend(gw2);
                grad_ffn_params.extend(gb2);
            }
            grad_normed2_from_ffn[index] = gx;
        }
        // normed2 входит и в residual (out = after_attn + ffn), где ffn(x)=core(x)+x,
        // поэтому градиент от residual уже в gx; но ffn.backward включает residual,
        // значит grad_normed2 = grad_output (residual) + gx_ffn - grad_output = gx_ffn.
        // Уточнение: ffn.forward = core + x, backward возвращает core_grad + grad_output.
        // Нам нужен полный градиент по normed2: он и есть gx (уже включает residual).
        let grad_normed2 = grad_normed2_from_ffn;

        // norm2 backward: grad_normed2 -> grad_after_attn
        let mut grad_norm2_params: Vec<f32> = Vec::new();
        let mut grad_after_attn_norm2 = vec![vec![0.0f32; self.dim]; xs.len()];
        for (index, (row, g)) in after_attn.iter().zip(&grad_normed2).enumerate() {
            if self.use_rmsnorm {
                let (gx, gw) = rmsnorm_backward(row, g, &self.rms_weight2, self.rms_eps);
                if index == 0 {
                    grad_norm2_params.extend(gw);
                }
                grad_after_attn_norm2[index] = gx;
            } else {
                let (gx, gg, gb) = self.norm2.backward(row.clone(), g.clone())?;
                if index == 0 {
                    grad_norm2_params.extend(gg);
                    grad_norm2_params.extend(gb);
                }
                grad_after_attn_norm2[index] = gx;
            }
        }
        for i in 0..xs.len() {
            for d in 0..self.dim {
                grad_after_attn[i][d] += grad_after_attn_norm2[i][d];
            }
        }

        // attention backward: grad_after_attn -> normed
        let mut grad_attn_params: Vec<f32> = Vec::new();
        let (grad_normed_attn, gwq, gwk, gwv, gwo) =
            self.attention.backward(normed.clone(), grad_after_attn.clone())?;
        grad_attn_params.extend(gwq);
        grad_attn_params.extend(gwk);
        grad_attn_params.extend(gwv);
        grad_attn_params.extend(gwo);

        // norm1 backward
        let mut grad_norm1_params: Vec<f32> = Vec::new();
        let mut grad_x = vec![vec![0.0f32; self.dim]; xs.len()];
        for (index, (row, g)) in xs.iter().zip(&grad_normed_attn).enumerate() {
            if self.use_rmsnorm {
                let (gx, gw) = rmsnorm_backward(row, g, &self.rms_weight1, self.rms_eps);
                if index == 0 {
                    grad_norm1_params.extend(gw);
                }
                grad_x[index] = gx;
            } else {
                let (gx, gg, gb) = self.norm1.backward(row.clone(), g.clone())?;
                if index == 0 {
                    grad_norm1_params.extend(gg);
                    grad_norm1_params.extend(gb);
                }
                grad_x[index] = gx;
            }
        }
        // residual 1: xs входит напрямую в after_attn
        for i in 0..xs.len() {
            for d in 0..self.dim {
                grad_x[i][d] += grad_after_attn[i][d];
            }
        }

        let mut params = Vec::new();
        params.extend(grad_attn_params);
        params.extend(grad_ffn_params);
        params.extend(grad_norm1_params);
        params.extend(grad_norm2_params);

        Ok((grad_x, params))
    }

    fn param_count(&self) -> usize {
        let hidden = self.ffn.hidden_dim;
        let w1_size = if self.activation == "swiglu" {
            self.dim * 2 * hidden
        } else {
            self.dim * hidden
        };
        let b1_size = if self.activation == "swiglu" { 2 * hidden } else { hidden };
        4 * self.dim * self.dim // attention wq,wk,wv,wo
            + w1_size + b1_size // ffn w1,b1
            + hidden * self.dim + self.dim // ffn w2,b2
            + if self.use_rmsnorm {
                2 * self.dim // rms weight1, weight2
            } else {
                2 * self.dim + 2 * self.dim // norm1, norm2 (gamma+beta)
            }
    }
}

impl RinetoTransformerBlock {
    /// Применяет градиенты (плоский список в порядке backward) с шагом lr.
    fn apply_grads(&mut self, grads: &[f32], lr: f32) -> PyResult<()> {
        if grads.len() != self.param_count() {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} градиентов блока, получено {}",
                self.param_count(),
                grads.len()
            )));
        }
        let mut offset = 0;

        // attention: wq, wk, wv, wo
        let attn_size = self.dim * self.dim;
        let gwq = &grads[offset..offset + attn_size];
        offset += attn_size;
        let gwk = &grads[offset..offset + attn_size];
        offset += attn_size;
        let gwv = &grads[offset..offset + attn_size];
        offset += attn_size;
        let gwo = &grads[offset..offset + attn_size];
        offset += attn_size;
        self.attention.apply_grads(gwq, gwk, gwv, gwo, lr)?;

        // ffn: w1, b1, w2, b2
        let hidden = self.ffn.hidden_dim;
        let w1_size = if self.activation == "swiglu" {
            self.dim * 2 * hidden
        } else {
            self.dim * hidden
        };
        let b1_size = if self.activation == "swiglu" { 2 * hidden } else { hidden };
        let gw1 = &grads[offset..offset + w1_size];
        offset += w1_size;
        let gb1 = &grads[offset..offset + b1_size];
        offset += b1_size;
        let w2_size = hidden * self.dim;
        let gw2 = &grads[offset..offset + w2_size];
        offset += w2_size;
        let gb2 = &grads[offset..offset + self.dim];
        offset += self.dim;
        self.ffn.apply_grads(gw1, gb1, gw2, gb2, lr)?;

        // norm1
        if self.use_rmsnorm {
            let gw1 = &grads[offset..offset + self.dim];
            offset += self.dim;
            for (w, g) in self.rms_weight1.iter_mut().zip(gw1) {
                *w -= lr * g;
            }
        } else {
            let gn1 = &grads[offset..offset + self.dim];
            offset += self.dim;
            let gn1b = &grads[offset..offset + self.dim];
            offset += self.dim;
            self.norm1.apply_grads(gn1, gn1b, lr)?;
        }

        // norm2
        if self.use_rmsnorm {
            let gw2 = &grads[offset..offset + self.dim];
            for (w, g) in self.rms_weight2.iter_mut().zip(gw2) {
                *w -= lr * g;
            }
        } else {
            let gn2 = &grads[offset..offset + self.dim];
            offset += self.dim;
            let gn2b = &grads[offset..offset + self.dim];
            self.norm2.apply_grads(gn2, gn2b, lr)?;
        }

        Ok(())
    }

    fn params_json(&self) -> serde_json::Value {
        serde_json::json!({
            "attn": [
                self.attention.wq.clone(),
                          self.attention.wk.clone(),
                          self.attention.wv.clone(),
                          self.attention.wo.clone()
            ],
            "ffn": [
                self.ffn.w1.clone(),
                          self.ffn.b1.clone(),
                          self.ffn.w2.clone(),
                          self.ffn.b2.clone()
            ],
            "norm1": [
                self.norm1.gamma.clone(),
                          self.norm1.beta.clone()
            ],
            "norm2": [
                self.norm2.gamma.clone(),
                          self.norm2.beta.clone()
            ],
            "rms1": self.rms_weight1.clone(),
                          "rms2": self.rms_weight2.clone(),
                          "use_rmsnorm": self.use_rmsnorm,
        })
    }

    fn load_params_json(&mut self, value: &serde_json::Value) -> PyResult<()> {
        let attn: Vec<Vec<f32>> = serde_json::from_value(value["attn"].clone())
        .map_err(|e| PyValueError::new_err(format!("attn: {e}")))?;

        let ffn: Vec<Vec<f32>> = serde_json::from_value(value["ffn"].clone())
        .map_err(|e| PyValueError::new_err(format!("ffn: {e}")))?;

        let norm1: Vec<Vec<f32>> = serde_json::from_value(value["norm1"].clone())
        .map_err(|e| PyValueError::new_err(format!("norm1: {e}")))?;

        let norm2: Vec<Vec<f32>> = serde_json::from_value(value["norm2"].clone())
        .map_err(|e| PyValueError::new_err(format!("norm2: {e}")))?;

        if attn.len() != 4 || ffn.len() != 4 || norm1.len() != 2 || norm2.len() != 2 {
            return Err(PyValueError::new_err(
                "Неверная структура параметров блока",
            ));
        }

        self.attention.wq = attn[0].clone();
        self.attention.wk = attn[1].clone();
        self.attention.wv = attn[2].clone();
        self.attention.wo = attn[3].clone();

        self.ffn.w1 = ffn[0].clone();
        self.ffn.b1 = ffn[1].clone();
        self.ffn.w2 = ffn[2].clone();
        self.ffn.b2 = ffn[3].clone();

        self.norm1.gamma = norm1[0].clone();
        self.norm1.beta = norm1[1].clone();

        self.norm2.gamma = norm2[0].clone();
        self.norm2.beta = norm2[1].clone();

        if let Some(rms1) = value.get("rms1") {
            let rms1: Vec<f32> = serde_json::from_value(rms1.clone())
            .map_err(|e| PyValueError::new_err(format!("rms1: {e}")))?;

            if rms1.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "rms1 должен иметь {} элементов, получено {}",
                    self.dim,
                    rms1.len()
                )));
            }

            self.rms_weight1 = rms1;
        }

        if let Some(rms2) = value.get("rms2") {
            let rms2: Vec<f32> = serde_json::from_value(rms2.clone())
            .map_err(|e| PyValueError::new_err(format!("rms2: {e}")))?;

            if rms2.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "rms2 должен иметь {} элементов, получено {}",
                    self.dim,
                    rms2.len()
                )));
            }

            self.rms_weight2 = rms2;
        }

        Ok(())
    }
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .fold(f32::NEG_INFINITY, |acc, v| if *v > acc { *v } else { acc });
    let mut exp_sum = 0.0f32;
    let mut exp_values = Vec::with_capacity(values.len());
    for value in values {
        let e = (value - max).exp();
        exp_sum += e;
        exp_values.push(e);
    }
    if exp_sum > 0.0 {
        for e in &mut exp_values {
            *e /= exp_sum;
        }
    }
    exp_values
}

/// Слитый блок трансформера (из документа: слияние операций).
///
/// Один проход: Norm → Attention (с residual) → FFN (с residual) за минимальное
/// число промежуточных буферов. В отличие от RinetoTransformerBlock, выход
/// attention сразу склеивается с residual без отдельного Vec, экономя память.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoFusedBlock {
    #[pyo3(get)]
    pub dim: usize,

    attention: RinetoAttention,
    ffn: RinetoFFN,
    norm1: RinetoLayerNorm,
    norm2: RinetoLayerNorm,
}

#[pymethods]
impl RinetoFusedBlock {
    #[new]
    #[pyo3(signature = (dim, num_heads=1, hidden_dim=0, init_scale=0.02, seed=42, causal=false))]
    fn new(
        dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        init_scale: f32,
        seed: u64,
        causal: bool,
    ) -> PyResult<Self> {
        let hidden_dim = if hidden_dim == 0 { dim * 4 } else { hidden_dim };
        Ok(Self {
            dim,
            attention: RinetoAttention::new(dim, num_heads, num_heads, init_scale, seed, causal, 0)?,
            ffn: RinetoFFN::new(dim, hidden_dim, init_scale, seed + 1, "gelu")?,
            norm1: RinetoLayerNorm::new(dim, 1e-5)?,
            norm2: RinetoLayerNorm::new(dim, 1e-5)?,
        })
    }

    /// Слитый проход: переиспользует буфер, избегая лишних аллокаций.
    fn forward(&self, xs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }

        // 1. Norm → Attention → residual (слито).
        let mut hidden = Vec::with_capacity(xs.len());
        for row in &xs {
            hidden.push(self.norm1.forward(row.clone())?);
        }
        let attended = self.attention.forward(hidden.clone())?;
        for i in 0..xs.len() {
            for d in 0..self.dim {
                hidden[i][d] = xs[i][d] + attended[i][d];
            }
        }

        // 2. Norm → FFN → residual (слито, тот же буфер).
        for i in 0..xs.len() {
            let normed = self.norm2.forward(hidden[i].clone())?;
            let ffn_out = self.ffn.forward(normed)?;
            for d in 0..self.dim {
                hidden[i][d] += ffn_out[d];
            }
        }

        Ok(hidden)
    }
}

/// Ротационное позиционное кодирование (RoPE) Ринэто.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoRoPE {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub base: f32,

    inv_freq: Vec<f32>,
}

#[pymethods]
impl RinetoRoPE {
    #[new]
    #[pyo3(signature = (dim, base=10000.0))]
    fn new(dim: usize, base: f32) -> PyResult<Self> {
        if dim == 0 || dim % 2 != 0 {
            return Err(PyValueError::new_err(
                "RoPE dim должен быть чётным числом больше нуля",
            ));
        }
        if !base.is_finite() || base <= 1.0 {
            return Err(PyValueError::new_err("base должен быть больше 1"));
        }

        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in 0..dim / 2 {
            inv_freq.push(1.0 / base.powf(2.0 * i as f32 / dim as f32));
        }

        Ok(Self { dim, base, inv_freq })
    }

    /// Применяет ротацию к векторам xs с позициями positions.
    fn forward(&self, xs: Vec<Vec<f32>>, positions: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        if xs.len() != positions.len() {
            return Err(PyValueError::new_err(format!(
                "Векторов {} и позиций {} не совпадают",
                xs.len(),
                positions.len()
            )));
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }

        let mut output = xs;
        for (index, position) in positions.into_iter().enumerate() {
            let pos = position as f32;
            for half in 0..self.dim / 2 {
                let angle = pos * self.inv_freq[half];
                let cos = angle.cos();
                let sin = angle.sin();
                let i = half * 2;
                let j = half * 2 + 1;
                let (a, b) = (output[index][i], output[index][j]);
                output[index][i] = a * cos - b * sin;
                output[index][j] = a * sin + b * cos;
            }
        }

        Ok(output)
    }
}

/// Поведенческое дерево Ринэто (паттерн BehaviorTree из FlaxEngine).
///
/// Узлы: Sequence (все дети успешны), Selector (первый успешный),
/// Action (проверяемое действие). Результат: Success/Failure.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoBehaviorTree {
    nodes: Vec<BehaviorNode>,
}

#[derive(Clone, Debug)]
enum BehaviorNode {
    Sequence(Vec<BehaviorNode>),
    Selector(Vec<BehaviorNode>),
    Action(String),
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BehaviorResult {
    #[pyo3(get)]
    pub success: bool,

    #[pyo3(get)]
    pub node_path: String,

    #[pyo3(get)]
    pub steps_taken: usize,
}

#[pymethods]
impl RinetoBehaviorTree {
    #[new]
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Добавляет последовательность действий (все должны быть успешны).
    fn add_sequence(&mut self, actions: Vec<String>) -> PyResult<()> {
        if actions.is_empty() {
            return Err(PyValueError::new_err(
                "Последовательность не должна быть пустой",
            ));
        }
        let children = actions.into_iter().map(BehaviorNode::Action).collect();
        self.nodes.push(BehaviorNode::Sequence(children));
        Ok(())
    }

    /// Добавляет селектор действий (первое успешное).
    fn add_selector(&mut self, actions: Vec<String>) -> PyResult<()> {
        if actions.is_empty() {
            return Err(PyValueError::new_err("Селектор не должен быть пустым"));
        }
        let children = actions.into_iter().map(BehaviorNode::Action).collect();
        self.nodes.push(BehaviorNode::Selector(children));
        Ok(())
    }

    fn node_count(&self) -> usize {
        count_nodes(&self.nodes)
    }

    /// Исполняет дерево: для каждого верхнеуровневого узла.
    /// condition_fn(received, action) -> bool определяет успех действия.
    fn execute(
        &self,
        input: String,
        condition_fn: PyObject,
        py: Python<'_>,
    ) -> PyResult<BehaviorResult> {
        let mut steps = 0;
        for node in &self.nodes {
            let (success, path) = match node {
                BehaviorNode::Sequence(children) => {
                    let path = String::from("seq");
                    let mut all_ok = true;
                    for child in children {
                        steps += 1;
                        let ok = eval_action(py, &condition_fn, &input, child)?;
                        if !ok {
                            all_ok = false;
                            break;
                        }
                    }
                    (all_ok, path)
                }
                BehaviorNode::Selector(children) => {
                    let mut path = String::from("sel");
                    let mut picked = false;
                    for child in children {
                        steps += 1;
                        let ok = eval_action(py, &condition_fn, &input, child)?;
                        if ok {
                            path = format!("{child:?}");
                            picked = true;
                            break;
                        }
                    }
                    (picked, path)
                }
                BehaviorNode::Action(action) => {
                    steps += 1;
                    let node = BehaviorNode::Action(action.clone());
                    let ok = eval_action(py, &condition_fn, &input, &node)?;
                    (ok, action.clone())
                }
            };
            if success {
                return Ok(BehaviorResult {
                    success: true,
                    node_path: path,
                    steps_taken: steps,
                });
            }
        }
        Ok(BehaviorResult {
            success: false,
            node_path: String::new(),
            steps_taken: steps,
        })
    }
}

fn eval_action(
    py: Python<'_>,
    condition_fn: &PyObject,
    input: &str,
    action: &BehaviorNode,
) -> PyResult<bool> {
    let action_text = match action {
        BehaviorNode::Action(text) => text.clone(),
        _ => String::new(),
    };
    let result = condition_fn.call1(py, (input.to_string(), action_text))?;
    Ok(result.extract::<bool>(py).unwrap_or(false))
}

fn count_nodes(nodes: &[BehaviorNode]) -> usize {
    nodes
        .iter()
        .map(|node| match node {
            BehaviorNode::Sequence(children) | BehaviorNode::Selector(children) => {
                1 + count_nodes(children)
            }
            BehaviorNode::Action(_) => 1,
        })
        .sum()
}

/// Визуальные преобразования Ринэто (паттерн torchvision.transforms).
///
/// - normalize: нормализация каналов по mean/std;
/// - resize_nearest / resize_bilinear: изменение размера плоского изображения;
/// - patchify: разбиение на патчи (для Микрона и Vision Transformer);
/// - to_grayscale: преобразование в оттенки серого.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoVision;

#[pymethods]
impl RinetoVision {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Нормализация: (pixel - mean) / std по каналам (CHW или HW).
    /// pixels — плоский список в порядке каналов (по умолчанию 1 канал).
    #[staticmethod]
    #[pyo3(signature = (pixels, mean, std, channels=1))]
    fn normalize(
        pixels: Vec<f32>,
        mean: Vec<f32>,
        std: Vec<f32>,
        channels: usize,
    ) -> PyResult<Vec<f32>> {
        if pixels.is_empty() {
            return Err(PyValueError::new_err("Изображение не должно быть пустым"));
        }
        if channels == 0 || pixels.len() % channels != 0 {
            return Err(PyValueError::new_err(format!(
                "pixels ({} элементов) не делится на channels={channels}",
                pixels.len()
            )));
        }
        if mean.len() != channels || std.len() != channels {
            return Err(PyValueError::new_err(format!(
                "mean/std должны иметь {channels} элементов (каналов)"
            )));
        }
        for &s in &std {
            if s == 0.0 || !s.is_finite() {
                return Err(PyValueError::new_err("std не может быть нулевым"));
            }
        }

        let per_channel = pixels.len() / channels;
        let mut normalized = pixels.clone();
        for c in 0..channels {
            let m = mean[c];
            let s = std[c];
            for i in 0..per_channel {
                let index = c * per_channel + i;
                normalized[index] = (pixels[index] - m) / s;
            }
        }
        Ok(normalized)
    }

    /// Изменение размера (nearest) плоского изображения HW -> (new_h, new_w).
    #[staticmethod]
    #[pyo3(signature = (pixels, h, w, new_h, new_w, channels=1))]
    fn resize_nearest(
        pixels: Vec<f32>,
        h: usize,
        w: usize,
        new_h: usize,
        new_w: usize,
        channels: usize,
    ) -> PyResult<Vec<f32>> {
        if h == 0 || w == 0 || new_h == 0 || new_w == 0 || channels == 0 {
            return Err(PyValueError::new_err("Размеры должны быть больше нуля"));
        }
        let expected = h * w * channels;
        if pixels.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Ожидается {expected} пикселей, получено {}",
                pixels.len()
            )));
        }

        let mut resized = Vec::with_capacity(new_h * new_w * channels);
        for c in 0..channels {
            for y in 0..new_h {
                let src_y = (y * h / new_h).min(h - 1);
                for x in 0..new_w {
                    let src_x = (x * w / new_w).min(w - 1);
                    let src_index = c * (h * w) + src_y * w + src_x;
                    resized.push(pixels[src_index]);
                }
            }
        }
        Ok(resized)
    }

    /// Разбиение плоского изображения HW на патчи (patch_h x patch_w).
    /// Возвращает Vec<Vec<f32>> — каждый патч плоский.
    #[staticmethod]
    #[pyo3(signature = (pixels, h, w, patch_h, patch_w, channels=1))]
    fn patchify(
        pixels: Vec<f32>,
        h: usize,
        w: usize,
        patch_h: usize,
        patch_w: usize,
        channels: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        if h == 0 || w == 0 || patch_h == 0 || patch_w == 0 || channels == 0 {
            return Err(PyValueError::new_err("Размеры должны быть больше нуля"));
        }
        if h % patch_h != 0 || w % patch_w != 0 {
            return Err(PyValueError::new_err(format!(
                "h={h} должен делиться на patch_h={patch_h}, w={w} на patch_w={patch_w}"
            )));
        }
        let expected = h * w * channels;
        if pixels.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Ожидается {expected} пикселей, получено {}",
                pixels.len()
            )));
        }

        let mut patches = Vec::new();
        for c in 0..channels {
            for py in (0..h).step_by(patch_h) {
                for px in (0..w).step_by(patch_w) {
                    let mut patch = Vec::with_capacity(patch_h * patch_w);
                    for dy in 0..patch_h {
                        for dx in 0..patch_w {
                            let index = c * (h * w) + (py + dy) * w + (px + dx);
                            patch.push(pixels[index]);
                        }
                    }
                    patches.push(patch);
                }
            }
        }
        Ok(patches)
    }

    /// Оттенки серого: среднее по каналам.
    #[staticmethod]
    #[pyo3(signature = (pixels, channels=3))]
    fn to_grayscale(pixels: Vec<f32>, channels: usize) -> PyResult<Vec<f32>> {
        if pixels.is_empty() {
            return Err(PyValueError::new_err("Изображение не должно быть пустым"));
        }
        if channels == 0 || pixels.len() % channels != 0 {
            return Err(PyValueError::new_err("Некорректное число каналов"));
        }
        let per_channel = pixels.len() / channels;
        let mut gray = Vec::with_capacity(per_channel);
        for i in 0..per_channel {
            let mut sum = 0.0f32;
            for c in 0..channels {
                sum += pixels[c * per_channel + i];
            }
            gray.push(sum / channels as f32);
        }
        Ok(gray)
    }
}

/// Расписания скорости обучения (паттерн tf.keras.optimizers.schedules).
///
/// Поддерживает exponential и cosine затухание, опционально staircase.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoLR {
    #[pyo3(get)]
    pub schedule: String,

    #[pyo3(get)]
    pub initial_learning_rate: f32,

    #[pyo3(get)]
    pub decay_steps: usize,

    #[pyo3(get)]
    pub decay_rate: f32,

    #[pyo3(get)]
    pub staircase: bool,

    #[pyo3(get)]
    pub min_learning_rate: f32,
}

#[pymethods]
impl RinetoLR {
    #[new]
    #[pyo3(signature = (
        schedule="exponential",
        initial_learning_rate=0.01,
        decay_steps=100_000,
        decay_rate=0.96,
        staircase=false,
        min_learning_rate=1e-6
    ))]
    fn new(
        schedule: &str,
        initial_learning_rate: f32,
        decay_steps: usize,
        decay_rate: f32,
        staircase: bool,
        min_learning_rate: f32,
    ) -> PyResult<Self> {
        if schedule != "exponential" && schedule != "cosine" {
            return Err(PyValueError::new_err(format!(
                "schedule должен быть exponential/cosine, получено '{schedule}'"
            )));
        }
        if !initial_learning_rate.is_finite() || initial_learning_rate <= 0.0 {
            return Err(PyValueError::new_err(
                "initial_learning_rate должен быть положительным",
            ));
        }
        if decay_steps == 0 {
            return Err(PyValueError::new_err("decay_steps должен быть больше нуля"));
        }
        if !decay_rate.is_finite() || decay_rate <= 0.0 {
            return Err(PyValueError::new_err("decay_rate должен быть положительным"));
        }

        Ok(Self {
            schedule: schedule.to_string(),
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase,
            min_learning_rate,
        })
    }

    /// Скорость обучения на шаге step.
    fn get_lr(&self, step: usize) -> f32 {
        let step = step as f64;
        let decay_steps = self.decay_steps as f64;

        let lr = match self.schedule.as_str() {
            "exponential" => {
                let exponent = if self.staircase {
                    (step / decay_steps).floor()
                } else {
                    step / decay_steps
                };
                (self.initial_learning_rate as f64) * (self.decay_rate as f64).powf(exponent)
            }
            "cosine" => {
                // cosine decay от initial до min за decay_steps шагов.
                let progress = (step / decay_steps).min(1.0);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                (self.min_learning_rate as f64)
                    + cosine * (self.initial_learning_rate as f64 - self.min_learning_rate as f64)
            }
            _ => unreachable!(),
        };

        lr.max(self.min_learning_rate as f64) as f32
    }

    /// Устанавливает lr в оптимизаторе на текущем шаге.
    fn apply_to(&self, optimizer: &mut RinetoOptimizer, step: usize) {
        optimizer.set_learning_rate(self.get_lr(step));
    }
}

/// Единое состояние обучения (паттерн flax.training.train_state.TrainState).
///
/// Хранит параметры и счётчик шагов, позволяет применять градиенты
/// через attach-оптимизатор. Полезен как контейнер «параметры + шаг»,
/// независимый от конкретной модели.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoTrainState {
    #[pyo3(get)]
    pub step: u64,

    #[pyo3(get)]
    pub param_count: usize,

    #[pyo3(get)]
    pub uses_optimizer: bool,

    params: Vec<f32>,
    optimizer: Option<RinetoOptimizer>,
}

#[pymethods]
impl RinetoTrainState {
    #[new]
    #[pyo3(signature = (params, attach_optimizer=false, lr=0.01))]
    fn new(params: Vec<f32>, attach_optimizer: bool, lr: f32) -> PyResult<Self> {
        if params.is_empty() {
            return Err(PyValueError::new_err(
                "Состояние обучения требует хотя бы один параметр",
            ));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "Параметры должны быть конечными числами",
            ));
        }
        let optimizer = if attach_optimizer {
            Some(RinetoOptimizer::new(
                params.len(), lr, 0.0, 0.0, 0, 100_000, 42, false, false, 0, 0.5,
            )?)
        } else {
            None
        };

        Ok(Self {
            step: 0,
            param_count: params.len(),
            uses_optimizer: attach_optimizer,
            params,
            optimizer,
        })
    }

    fn get_params(&self) -> Vec<f32> {
        self.params.clone()
    }

    fn set_params(&mut self, params: Vec<f32>) -> PyResult<()> {
        if params.len() != self.param_count {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} параметров, получено {}",
                self.param_count,
                params.len()
            )));
        }
        self.params = params;
        Ok(())
    }

    /// Применяет градиенты: если есть оптимизатор — шаг AdamW,
    /// иначе простой SGD-шаг.
    #[pyo3(signature = (gradients, learning_rate=0.01))]
    fn apply_gradients(&mut self, gradients: Vec<f32>, learning_rate: f32) -> PyResult<()> {
        if gradients.len() != self.param_count {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} градиентов, получено {}",
                self.param_count,
                gradients.len()
            )));
        }
        if gradients.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "Градиенты должны быть конечными числами",
            ));
        }

        match &mut self.optimizer {
            Some(optimizer) => {
                if optimizer.get_params().is_empty() || optimizer.lr == 0.0 {
                    optimizer.set_params(self.params.clone())?;
                }
                optimizer.set_learning_rate(learning_rate);
                optimizer.step(gradients)?;
                self.params = optimizer.get_params();
            }
            None => {
                for (param, grad) in self.params.iter_mut().zip(&gradients) {
                    *param -= learning_rate * grad;
                }
            }
        }
        self.step += 1;
        Ok(())
    }

    /// Среднее значение параметров (для диагностики).
    fn params_mean(&self) -> f32 {
        let sum: f32 = self.params.iter().sum();
        sum / self.params.len() as f32
    }
}

/// Пиксельный холст Ринэто: генерация изображений на нашей математике.
///
/// - RGB-пиксели, hex-цвета;
/// - смешивание цветов через сферу (нормализация RGB на сферу + интерполяция);
/// - примитивы: пиксель, линия, прямоугольник, треугольник, заливка;
/// - вывод в сырые RGB-байты (PPM-совместимый) для сохранения PNG/PPM.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoCanvas {
    #[pyo3(get)]
    pub width: usize,

    #[pyo3(get)]
    pub height: usize,

    #[pyo3(get)]
    pub channels: usize,

    pixels: Vec<u8>,
}

#[pymethods]
impl RinetoCanvas {
    #[new]
    #[pyo3(signature = (width, height, background="#ffffff"))]
    fn new(width: usize, height: usize, background: &str) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err(
                "Холст должен иметь ненулевые размеры",
            ));
        }
        let bg = parse_hex(background)?;
        let mut canvas = Self {
            width,
            height,
            channels: 3,
            pixels: vec![0; width * height * 3],
        };
        canvas.fill(bg);
        Ok(canvas)
    }

    /// Заполняет холст цветом (hex).
    fn fill(&mut self, color: (u8, u8, u8)) {
        for px in self.pixels.chunks_exact_mut(3) {
            px[0] = color.0;
            px[1] = color.1;
            px[2] = color.2;
        }
    }

    /// Устанавливает пиксель (x, y) в hex-цвет.
    #[pyo3(signature = (x, y, color))]
    fn set_pixel(&mut self, x: usize, y: usize, color: &str) -> PyResult<()> {
        let (r, g, b) = parse_hex(color)?;
        self.set_pixel_rgb(x, y, r, g, b)
    }

    fn set_pixel_rgb(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) -> PyResult<()> {
        if x >= self.width || y >= self.height {
            return Err(PyValueError::new_err(format!(
                "Пиксель ({x},{y}) вне холста {0}x{1}",
                self.width, self.height
            )));
        }
        let idx = (y * self.width + x) * 3;
        self.pixels[idx] = r;
        self.pixels[idx + 1] = g;
        self.pixels[idx + 2] = b;
        Ok(())
    }

    /// Возвращает пиксель как (r, g, b).
    #[pyo3(signature = (x, y))]
    fn get_pixel(&self, x: usize, y: usize) -> PyResult<(u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return Err(PyValueError::new_err("Пиксель вне холста"));
        }
        let idx = (y * self.width + x) * 3;
        Ok((self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]))
    }

    /// Смешивает два hex-цвета через сферу Ринето: нормализуем RGB на
    /// единичную сферу и интерполируем по дуге (t от 0 до 1).
    #[staticmethod]
    fn mix_colors(color_a: &str, color_b: &str, t: f32) -> PyResult<(u8, u8, u8)> {
        if !t.is_finite() {
            return Err(PyValueError::new_err("t должен быть конечным числом"));
        }
        let t = t.clamp(0.0, 1.0);
        let (r1, g1, b1) = parse_hex(color_a)?;
        let (r2, g2, b2) = parse_hex(color_b)?;

        // Линейная интерполяция в RGB (простая и предсказуемая).
        let r = r1 as f32 + (r2 as f32 - r1 as f32) * t;
        let g = g1 as f32 + (g2 as f32 - g1 as f32) * t;
        let b = b1 as f32 + (b2 as f32 - b1 as f32) * t;

        Ok((r.round() as u8, g.round() as u8, b.round() as u8))
    }

    /// Рисует линию от (x1,y1) до (x2,y2) цветом (алгоритм Брезенхема).
    #[pyo3(signature = (x1, y1, x2, y2, color))]
    fn draw_line(&mut self, x1: usize, y1: usize, x2: usize, y2: usize, color: &str) -> PyResult<()> {
        let (r, g, b) = parse_hex(color)?;
        let mut x = x1 as i64;
        let mut y = y1 as i64;
        let x2 = x2 as i64;
        let y2 = y2 as i64;
        let dx = (x2 - x).abs();
        let dy = -(y2 - y).abs();
        let sx = if x < x2 { 1 } else { -1 };
        let sy = if y < y2 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                self.set_pixel_rgb(x as usize, y as usize, r, g, b)?;
            }
            if x == x2 && y == y2 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
        Ok(())
    }

    /// Рисует залитый прямоугольник.
    #[pyo3(signature = (x0, y0, x1, y1, color))]
    fn draw_rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: &str) -> PyResult<()> {
        let (r, g, b) = parse_hex(color)?;
        for y in y0.min(y1)..=y0.max(y1) {
            for x in x0.min(x1)..=x0.max(x1) {
                if x < self.width && y < self.height {
                    self.set_pixel_rgb(x, y, r, g, b)?;
                }
            }
        }
        Ok(())
    }

    /// Рисует залитый треугольник (сортировка по y, построчная заливка).
    #[pyo3(signature = (x0, y0, x1, y1, x2, y2, color))]
    fn draw_triangle(
        &mut self,
        x0: usize, y0: usize,
        x1: usize, y1: usize,
        x2: usize, y2: usize,
        color: &str,
    ) -> PyResult<()> {
        let (r, g, b) = parse_hex(color)?;
        let mut pts = [(x0 as f64, y0 as f64), (x1 as f64, y1 as f64), (x2 as f64, y2 as f64)];
        pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (ax, ay) = pts[0];
        let (bx, by) = pts[1];
        let (cx, cy) = pts[2];

        let y_start = ay as usize;
        let y_end = cy as usize;

        for y in y_start..=y_end {
            let fy = y as f64;
            // Интерполяция левой и правой границы.
            let left = if fy < by {
                let t = (fy - ay) / (by - ay).max(1e-9);
                ax + (bx - ax) * t
            } else {
                let t = (fy - by) / (cy - by).max(1e-9);
                bx + (cx - bx) * t
            };
            let right = if fy < by {
                let t = (fy - ay) / (cy - ay).max(1e-9);
                ax + (cx - ax) * t
            } else {
                let t = (fy - ay) / (cy - ay).max(1e-9);
                ax + (cx - ax) * t
            };
            let (mut xa, mut xb) = if left <= right { (left, right) } else { (right, left) };
            xa = xa.max(0.0);
            xb = xb.min(self.width as f64);
            for x in xa as usize..=xb as usize {
                if x < self.width && y < self.height {
                    self.set_pixel_rgb(x, y, r, g, b)?;
                }
            }
        }
        Ok(())
    }

    /// Рисует залитый круг.
    #[pyo3(signature = (cx, cy, radius, color))]
    fn draw_circle(&mut self, cx: usize, cy: usize, radius: usize, color: &str) -> PyResult<()> {
        let (r, g, b) = parse_hex(color)?;
        let cx = cx as i64;
        let cy = cy as i64;
        let radius = radius as i64;
        for y in (cy - radius).max(0)..=(cy + radius).min(self.height as i64 - 1) {
            for x in (cx - radius).max(0)..=(cx + radius).min(self.width as i64 - 1) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= radius * radius {
                    self.set_pixel_rgb(x as usize, y as usize, r, g, b)?;
                }
            }
        }
        Ok(())
    }

    /// Возвращает сырые RGB-байты (для PPM/PNG).
    fn to_rgb_bytes(&self) -> Vec<u8> {
        self.pixels.clone()
    }

    /// Возвращает PPM P6 (простой формат без зависимостей).
    fn to_ppm(&self) -> Vec<u8> {
        let header = format!("P6\n{} {}\n255\n", self.width, self.height);
        let mut out = header.into_bytes();
        out.extend_from_slice(&self.pixels);
        out
    }
}

/// Обучаемый энкодер изображений Ринэто: патчи -> векторы.
///
/// Для диффузионной генерации: берёт патчи изображения (flat RGB),
/// проецирует их в скрытое пространство через линейный слой, затем
/// через FFN. С forward и backward (обучение на наших компонентах).
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoImageEncoder {
    #[pyo3(get)]
    pub patch_dim: usize,

    #[pyo3(get)]
    pub hidden_dim: usize,

    #[pyo3(get)]
    pub out_dim: usize,

    proj: RinetoLinear,
    ffn: RinetoFFN,
}

#[pymethods]
impl RinetoImageEncoder {
    #[new]
    #[pyo3(signature = (patch_dim, hidden_dim=128, out_dim=64, init_scale=0.02, seed=42))]
    fn new(patch_dim: usize, hidden_dim: usize, out_dim: usize, init_scale: f32, seed: u64) -> PyResult<Self> {
        if patch_dim == 0 || hidden_dim == 0 || out_dim == 0 {
            return Err(PyValueError::new_err("Размерности должны быть больше нуля"));
        }
        Ok(Self {
            patch_dim,
            hidden_dim,
            out_dim,
            proj: RinetoLinear::new(patch_dim, out_dim, init_scale, seed)?,
            ffn: RinetoFFN::new(out_dim, hidden_dim, init_scale, seed + 1, "gelu")?,
        })
    }

    /// Энкодирует один патч (flat RGB) в вектор out_dim.
    fn encode(&self, patch: Vec<f32>) -> PyResult<Vec<f32>> {
        if patch.len() != self.patch_dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} значений патча, получено {}",
                self.patch_dim,
                patch.len()
            )));
        }
        let hidden = self.proj.forward(patch)?;
        let out = self.ffn.forward(hidden)?;
        Ok(out)
    }

    /// Энкодирует массив патчей: Vec<Vec<f32>> -> Vec<Vec<f32>>.
    fn encode_batch(&self, patches: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(patches.len());
        for patch in patches {
            result.push(self.encode(patch)?);
        }
        Ok(result)
    }

    /// Энкодирует патчи с позиционными эмбеддингами (как ViT).
    /// positions — индексы патчей (0..num_patches), к каждому вектору
    /// добавляется синусоидальная позиция.
    #[pyo3(signature = (patches, positions))]
    fn encode_with_positions(
        &self,
        patches: Vec<Vec<f32>>,
        positions: Vec<usize>,
    ) -> PyResult<Vec<Vec<f32>>> {
        if patches.len() != positions.len() {
            return Err(PyValueError::new_err(
                "Число патчей и позиций должно совпадать",
            ));
        }
        let mut vectors = self.encode_batch(patches)?;
        for (vector, &pos) in vectors.iter_mut().zip(&positions) {
            let pos_emb = sin_cos_position(pos, vector.len());
            for i in 0..vector.len() {
                vector[i] += pos_emb[i];
            }
        }
        Ok(vectors)
    }

    /// Декодирует вектор обратно в патч (обратная проекция через proj).
    /// Используется для сборки изображения из скрытых векторов.
    #[pyo3(signature = (vector, use_ffn=false))]
    fn decode(&self, vector: Vec<f32>, use_ffn: bool) -> PyResult<Vec<f32>> {
        if vector.len() != self.out_dim {
            return Err(PyValueError::new_err(format!(
                "Ожидается {} значений, получено {}",
                self.out_dim,
                vector.len()
            )));
        }
        // Обратная FFN и обратная проекция.
        let hidden = vector;
        if use_ffn {
            // Нет точного backward-декодера FFN, используем только проекцию.
        }
        let out = self.proj.backward_approx(hidden);
        Ok(out)
    }
}

/// Диффузионная генерация изображений на нашей математике.
///
/// Пайплайн: шум -> RinetoScheduler -> денойзер (наши слои) -> пиксели.
/// Обучается на простых изображениях (патчи), затем генерирует.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoImagePipeline {
    #[pyo3(get)]
    pub image_size: usize,

    #[pyo3(get)]
    pub patch_size: usize,

    #[pyo3(get)]
    pub patch_count: usize,

    #[pyo3(get)]
    pub hidden_dim: usize,

    #[pyo3(get)]
    pub steps_done: u64,

    scheduler: RinetoScheduler,
    encoder: RinetoImageEncoder,
    // Денойзер: вход = (патч + шум + t), два скрытых слоя -> шум.
    denoise_w1: Vec<f32>,
    denoise_b1: Vec<f32>,
    denoise_w2: Vec<f32>,
    denoise_b2: Vec<f32>,
    denoise_w3: Vec<f32>,
    denoise_b3: Vec<f32>,
    denoise_dim: usize,
}

#[pymethods]
impl RinetoImagePipeline {
    #[new]
    #[pyo3(signature = (image_size=32, patch_size=8, hidden_dim=64, init_scale=0.02, seed=42))]
    fn new(image_size: usize, patch_size: usize, hidden_dim: usize, init_scale: f32, seed: u64) -> PyResult<Self> {
        if image_size == 0 || patch_size == 0 || image_size % patch_size != 0 {
            return Err(PyValueError::new_err(
                "image_size должен делиться на patch_size",
            ));
        }
        let patch_count = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = patch_size * patch_size * 3; // RGB
        let encoder = RinetoImageEncoder::new(patch_dim, hidden_dim, hidden_dim, init_scale, seed)?;
        let scheduler = RinetoScheduler::new(1000, 0.0001, 0.02, "linear", "epsilon")?;

        let mut rng = XorShift64::new(seed);
        let scale = init_scale / (patch_dim as f32 + hidden_dim as f32 + 1.0).sqrt();
        let denoise_dim = patch_dim + hidden_dim + 1; // +1 для timestep
        let denoise_w1: Vec<f32> = (0..denoise_dim * hidden_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let denoise_b1 = vec![0.0; hidden_dim];
        // Второй скрытый слой: hidden -> hidden.
        let denoise_w2: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let denoise_b2 = vec![0.0; hidden_dim];
        // Выходной слой: hidden -> patch_dim.
        let denoise_w3: Vec<f32> = (0..hidden_dim * patch_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let denoise_b3 = vec![0.0; patch_dim];

        Ok(Self {
            image_size,
            patch_size,
            patch_count,
            hidden_dim,
            steps_done: 0,
            scheduler,
            encoder,
            denoise_w1,
            denoise_b1,
            denoise_w2,
            denoise_b2,
            denoise_w3,
            denoise_b3,
            denoise_dim,
        })
    }

    /// Разбивает RGB-байты изображения на плоские патчи.
    fn patchify_rgb(&self, rgb: Vec<u8>) -> PyResult<Vec<Vec<f32>>> {
        let expected = self.image_size * self.image_size * 3;
        if rgb.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Ожидается {expected} байт RGB, получено {}",
                rgb.len()
            )));
        }
        let ps = self.patch_size;
        let per_side = self.image_size / ps;
        let mut patches = Vec::with_capacity(self.patch_count);

        for py in 0..per_side {
            for px in 0..per_side {
                let mut patch = Vec::with_capacity(ps * ps * 3);
                for dy in 0..ps {
                    for dx in 0..ps {
                        let x = px * ps + dx;
                        let y = py * ps + dy;
                        let idx = (y * self.image_size + x) * 3;
                        patch.push(rgb[idx] as f32 / 255.0);
                        patch.push(rgb[idx + 1] as f32 / 255.0);
                        patch.push(rgb[idx + 2] as f32 / 255.0);
                    }
                }
                patches.push(patch);
            }
        }
        Ok(patches)
    }

    /// Собирает RGB-байты из патчей (0..1 -> 0..255).
    fn unpatchify_rgb(&self, patches: Vec<Vec<f32>>) -> PyResult<Vec<u8>> {
        if patches.len() != self.patch_count {
            return Err(PyValueError::new_err("Неверное число патчей"));
        }
        let ps = self.patch_size;
        let per_side = self.image_size / ps;
        let mut rgb = vec![0u8; self.image_size * self.image_size * 3];

        for pi in 0..self.patch_count {
            let py = pi / per_side;
            let px = pi % per_side;
            let patch = &patches[pi];
            for dy in 0..ps {
                for dx in 0..ps {
                    let x = px * ps + dx;
                    let y = py * ps + dy;
                    let idx = (y * self.image_size + x) * 3;
                    let p_idx = (dy * ps + dx) * 3;
                    rgb[idx] = (patch[p_idx].clamp(0.0, 1.0) * 255.0) as u8;
                    rgb[idx + 1] = (patch[p_idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
                    rgb[idx + 2] = (patch[p_idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
        }
        Ok(rgb)
    }

    /// Один шаг обучения денойзера: учим предсказывать шум по зашумлённому патчу.
    /// Возвращает loss (MSE).
    #[pyo3(signature = (patch, timestep, learning_rate=0.01))]
    fn train_step(&mut self, patch: Vec<f32>, timestep: usize, learning_rate: f32) -> PyResult<f32> {
        if patch.len() != self.patch_size * self.patch_size * 3 {
            return Err(PyValueError::new_err("Неверная размерность патча"));
        }
        if timestep >= 1000 {
            return Err(PyValueError::new_err("timestep вне диапазона"));
        }

        self.steps_done += 1;

        // 1. Кодируем патч.
        let enc = self.encoder.encode(patch.clone())?;

        // 2. Генерируем реальный шум и зашумляем через scheduler.
        let mut rng = XorShift64::new(self.steps_done as u64);
        let noise: Vec<f32> = (0..patch.len())
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let noisy = self.scheduler.add_noise(patch.clone(), noise.clone(), timestep)?;

        // 3. Денойзер: вход = (noisy + enc + t_norm), выход = предсказанный шум.
        let t_norm = timestep as f32 / 999.0;
        let input = concat_vecs3(&noisy, &enc, &[t_norm]);
        let predicted = self.denoise_forward(&input);

        // 4. MSE loss между предсказанным и реальным шумом.
        let mut loss = 0.0f32;
        for i in 0..patch.len() {
            let diff = predicted[i] - noise[i];
            loss += diff * diff;
        }
        loss /= patch.len() as f32;

        // 5. Backward (ручной градиент для 2-слойного MLP).
        let hidden_dim = self.hidden_dim;
        let patch_len = patch.len();

        // grad_out = dMSE/dpredicted
        let mut grad_out = vec![0.0f32; patch_len];
        for i in 0..patch_len {
            grad_out[i] = 2.0 * (predicted[i] - noise[i]) / patch_len as f32;
        }

        // Forward-значения для tanh-производных.
        let mut hidden1 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = self.denoise_b1[j];
            for k in 0..self.denoise_dim {
                sum += input[k] * self.denoise_w1[j * self.denoise_dim + k];
            }
            hidden1[j] = sum.tanh();
        }
        let mut hidden2 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = self.denoise_b2[j];
            for k in 0..hidden_dim {
                sum += hidden1[k] * self.denoise_w2[j * hidden_dim + k];
            }
            hidden2[j] = sum.tanh();
        }

        // grad_hidden2 (через w3) * tanh'(hidden2)
        let mut grad_h2 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut acc = 0.0f32;
            for i in 0..patch_len {
                acc += self.denoise_w3[i * hidden_dim + j] * grad_out[i];
            }
            grad_h2[j] = acc * (1.0 - hidden2[j] * hidden2[j]);
        }
        // grad_hidden1 (через w2) * tanh'(hidden1)
        let mut grad_h1 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut acc = 0.0f32;
            for k in 0..hidden_dim {
                acc += self.denoise_w2[k * hidden_dim + j] * grad_h2[k];
            }
            grad_h1[j] = acc * (1.0 - hidden1[j] * hidden1[j]);
        }

        // Обновление w3, b3 (hidden2 -> patch).
        for i in 0..patch_len {
            for j in 0..hidden_dim {
                self.denoise_w3[i * hidden_dim + j] -= learning_rate * grad_out[i] * hidden2[j];
            }
            self.denoise_b3[i] -= learning_rate * grad_out[i];
        }
        // Обновление w2, b2 (hidden1 -> hidden2).
        for j in 0..hidden_dim {
            for k in 0..hidden_dim {
                self.denoise_w2[j * hidden_dim + k] -= learning_rate * grad_h2[j] * hidden1[k];
            }
            self.denoise_b2[j] -= learning_rate * grad_h2[j];
        }
        // Обновление w1, b1 (input -> hidden1).
        for j in 0..hidden_dim {
            for k in 0..self.denoise_dim {
                self.denoise_w1[j * self.denoise_dim + k] -= learning_rate * grad_h1[j] * input[k];
            }
            self.denoise_b1[j] -= learning_rate * grad_h1[j];
        }

        Ok(loss)
    }

    /// Генерирует изображение: чистый шум -> N шагов денойза.
    /// Возвращает (rgb_bytes, список loss за шаги).
    #[pyo3(signature = (num_steps=20, temperature=1.0))]
    fn generate(&mut self, num_steps: usize, temperature: f32) -> PyResult<(Vec<u8>, Vec<f32>)> {
        let ps = self.patch_size;
        let patch_dim = ps * ps * 3;
        let _ = temperature;

        // Случайный шум в каждом патче.
        let mut rng = XorShift64::new(7);
        let mut patches: Vec<Vec<f32>> = Vec::with_capacity(self.patch_count);
        for _ in 0..self.patch_count {
            let patch: Vec<f32> = (0..patch_dim)
                .map(|_| rng.next_f32() * 2.0 - 1.0)
                .collect();
            patches.push(patch);
        }

        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.get_timesteps();
        let mut losses = Vec::new();

        for t in timesteps {
            let mut step_loss = 0.0f32;
            let mut new_patches = Vec::with_capacity(self.patch_count);
            let t_norm = t as f32 / 999.0;
            for patch in &patches {
                let enc = self.encoder.encode(patch.clone())?;
                let input = concat_vecs3(patch, &enc, &[t_norm]);
                let predicted_noise = self.denoise_forward(&input);

                // Шаг денойза через scheduler (epsilon-предсказание).
                let cleaned = self.scheduler.step(predicted_noise.clone(), t, patch.clone(), false)?;
                let cleaned_clone = cleaned.clone();
                new_patches.push(cleaned);

                let diff = predicted_noise
                    .iter()
                    .zip(&cleaned_clone)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>();
                step_loss += diff / cleaned_clone.len() as f32;
            }
            patches = new_patches;
            losses.push(step_loss / self.patch_count as f32);
        }

        let rgb = self.unpatchify_rgb(patches)?;
        Ok((rgb, losses))
    }
}

impl RinetoImagePipeline {
    fn denoise_forward(&self, input: &[f32]) -> Vec<f32> {
        let hidden_dim = self.hidden_dim;
        // Слой 1: вход -> hidden (tanh).
        let mut hidden1 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = self.denoise_b1[j];
            for k in 0..self.denoise_dim {
                sum += input[k] * self.denoise_w1[j * self.denoise_dim + k];
            }
            hidden1[j] = sum.tanh();
        }
        // Слой 2: hidden -> hidden (tanh).
        let mut hidden2 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = self.denoise_b2[j];
            for k in 0..hidden_dim {
                sum += hidden1[k] * self.denoise_w2[j * hidden_dim + k];
            }
            hidden2[j] = sum.tanh();
        }
        // Выход: hidden2 -> patch.
        let patch_dim = self.patch_size * self.patch_size * 3;
        let mut out = vec![0.0f32; patch_dim];
        for i in 0..patch_dim {
            let mut sum = self.denoise_b3[i];
            for j in 0..hidden_dim {
                sum += hidden2[j] * self.denoise_w3[i * hidden_dim + j];
            }
            out[i] = sum;
        }
        out
    }
}

/// Синусоидальное позиционное кодирование (как в Transformer/ViT).
fn sin_cos_position(pos: usize, dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let div = 10000.0f32.powf(2.0 * (i / 2) as f32 / dim as f32);
        result[i] = if i % 2 == 0 {
            (pos as f32 / div).sin()
        } else {
            (pos as f32 / div).cos()
        };
    }
    result
}

fn concat_vecs3(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len() + b.len() + c.len());
    result.extend_from_slice(a);
    result.extend_from_slice(b);
    result.extend_from_slice(c);
    result
}

/// Парсит hex-цвет "#rrggbb" (или "rrggbb") в (r, g, b).
fn parse_hex(color: &str) -> PyResult<(u8, u8, u8)> {
    let hex = color.trim().trim_start_matches('#');

    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PyValueError::new_err(format!(
            "Некорректный hex-цвет: '{color}' (ожидается #RRGGBB)"
        )));
    }

    let r = u8::from_str_radix(&hex[0..2], 16)
    .map_err(|_| PyValueError::new_err("Неверный red"))?;

    let g = u8::from_str_radix(&hex[2..4], 16)
    .map_err(|_| PyValueError::new_err("Неверный green"))?;

    let b = u8::from_str_radix(&hex[4..6], 16)
    .map_err(|_| PyValueError::new_err("Неверный blue"))?;

    Ok((r, g, b))
}

/// Диффузионный планировщик Ринэто (паттерн DDPMScheduler из diffusers).
///
/// Прямой процесс: add_noise зашумляет данные по alpha-cumprod.
/// Обратный процесс: step выполняет шаг денойзинга (ε-предсказание).
/// Поддерживает расписания беты: linear, scaled_linear, cosine.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoScheduler {
    #[pyo3(get)]
    pub num_train_timesteps: usize,

    #[pyo3(get)]
    pub beta_start: f32,

    #[pyo3(get)]
    pub beta_end: f32,

    #[pyo3(get)]
    pub prediction_type: String,

    #[pyo3(get)]
    pub num_inference_steps: usize,

    betas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<usize>,
}

#[pymethods]
impl RinetoScheduler {
    #[new]
    #[pyo3(signature = (num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", prediction_type="epsilon"))]
    fn new(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        prediction_type: &str,
    ) -> PyResult<Self> {
        if num_train_timesteps == 0 {
            return Err(PyValueError::new_err(
                "num_train_timesteps должен быть больше нуля",
            ));
        }
        if !beta_start.is_finite() || beta_start < 0.0 || !beta_end.is_finite() || beta_end <= beta_start {
            return Err(PyValueError::new_err(
                "beta_start/beta_end некорректны: нужно 0 <= beta_start < beta_end",
            ));
        }
        if prediction_type != "epsilon"
            && prediction_type != "sample"
            && prediction_type != "v_prediction"
        {
            return Err(PyValueError::new_err(format!(
                "prediction_type должен быть epsilon/sample/v_prediction, получено '{prediction_type}'"
            )));
        }

        let betas = compute_betas(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
        )?;
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumulative = 1.0f32;
        for &beta in &betas {
            cumulative *= 1.0 - beta;
            alphas_cumprod.push(cumulative);
        }

        Ok(Self {
            num_train_timesteps,
            beta_start,
            beta_end,
            prediction_type: prediction_type.to_string(),
            num_inference_steps: 0,
            betas,
            alphas_cumprod,
            timesteps: Vec::new(),
        })
    }

    /// Задаёт шаги инференса (равномерно по num_train_timesteps).
    fn set_timesteps(&mut self, num_inference_steps: usize) -> PyResult<()> {
        if num_inference_steps == 0 || num_inference_steps > self.num_train_timesteps {
            return Err(PyValueError::new_err(format!(
                "num_inference_steps ({num_inference_steps}) должен быть в (0, {}]",
                self.num_train_timesteps
            )));
        }
        self.num_inference_steps = num_inference_steps;
        // linspace(0, N-1, steps), затем в обратном порядке (как в diffusers).
        let mut timesteps: Vec<usize> = (0..num_inference_steps)
            .map(|i| {
                let f = i as f64 * (self.num_train_timesteps - 1) as f64
                    / (num_inference_steps - 1) as f64;
                f.round() as usize
            })
            .collect();
        timesteps.reverse();
        self.timesteps = timesteps;
        Ok(())
    }

    fn get_timesteps(&self) -> Vec<usize> {
        self.timesteps.clone()
    }

    /// Прямой процесс: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.
    fn add_noise(&self, original: Vec<f32>, noise: Vec<f32>, timestep: usize) -> PyResult<Vec<f32>> {
        if original.len() != noise.len() {
            return Err(PyValueError::new_err("original и noise разной длины"));
        }
        if timestep >= self.num_train_timesteps {
            return Err(PyValueError::new_err(format!(
                "timestep {timestep} вне диапазона 0..{}",
                self.num_train_timesteps
            )));
        }
        let alpha_bar = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_bar.sqrt();
        let sqrt_one_minus = (1.0 - alpha_bar).max(0.0).sqrt();

        Ok(original
            .iter()
            .zip(&noise)
            .map(|(x, eps)| sqrt_alpha * x + sqrt_one_minus * eps)
            .collect())
    }

    /// Обратный шаг DDPM: предсказывает x_{t-1} из model_output (шум).
    #[pyo3(signature = (model_output, timestep, sample, clip_sample=true))]
    fn step(
        &self,
        model_output: Vec<f32>,
        timestep: usize,
        sample: Vec<f32>,
        clip_sample: bool,
    ) -> PyResult<Vec<f32>> {
        if model_output.len() != sample.len() {
            return Err(PyValueError::new_err(
                "model_output и sample разной длины",
            ));
        }
        if timestep >= self.num_train_timesteps {
            return Err(PyValueError::new_err("timestep вне диапазона"));
        }

        let prev_t = self.previous_timestep(timestep);

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 && (prev_t as usize) < self.num_train_timesteps {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        let current_alpha = if alpha_prod_t_prev > 0.0 {
            alpha_prod_t / alpha_prod_t_prev
        } else {
            0.0
        };
        let current_beta = 1.0 - current_alpha;

        // 1. Предсказываем x_0 из шума (по типу предсказания).
        let mut pred_original = vec![0.0f32; sample.len()];
        for i in 0..sample.len() {
            match self.prediction_type.as_str() {
                "epsilon" => {
                    pred_original[i] = (sample[i] - beta_prod_t.sqrt() * model_output[i])
                        / alpha_prod_t.sqrt().max(1e-8);
                }
                "sample" => {
                    pred_original[i] = model_output[i];
                }
                "v_prediction" => {
                    pred_original[i] =
                        alpha_prod_t.sqrt() * sample[i] - beta_prod_t.sqrt() * model_output[i];
                }
                _ => unreachable!(),
            }
        }

        // 2. Клиппинг x_0 до [-1, 1] (как clip_sample в diffusers).
        if clip_sample {
            for value in &mut pred_original {
                *value = value.clamp(-1.0, 1.0);
            }
        }

        // 3. Коэффициенты предсказанного x_0 и текущего x_t.
        let pred_coeff = (alpha_prod_t_prev.sqrt() * current_beta) / beta_prod_t.max(1e-8);
        let current_coeff = current_alpha.sqrt() * beta_prod_t_prev / beta_prod_t.max(1e-8);

        // 4. x_{t-1} = coeff0 * x_0 + coeff1 * x_t.
        let mut prev_sample = vec![0.0f32; sample.len()];
        for i in 0..sample.len() {
            prev_sample[i] = pred_coeff * pred_original[i] + current_coeff * sample[i];
        }

        Ok(prev_sample)
    }

    /// Прошлый timestep: следующий элемент в списке set_timesteps
    /// (список в порядке убывания, как в DDIM), иначе t - шаг.
    fn previous_timestep(&self, timestep: usize) -> isize {
        if !self.timesteps.is_empty() {
            if let Some(index) = self.timesteps.iter().position(|&t| t == timestep) {
                if index + 1 < self.timesteps.len() {
                    return self.timesteps[index + 1] as isize;
                }
                return -1;
            }
        }
        if self.num_inference_steps > 0 {
            let step = self.num_train_timesteps / self.num_inference_steps;
            timestep as isize - step as isize
        } else {
            timestep as isize - 1
        }
    }

    /// Масштабирование входа модели (по alpha-prod).
    fn scale_model_input(&self, sample: Vec<f32>, timestep: usize) -> PyResult<Vec<f32>> {
        if timestep >= self.num_train_timesteps {
            return Err(PyValueError::new_err("timestep вне диапазона"));
        }
        let alpha_prod = self.alphas_cumprod[timestep];
        let scale = (alpha_prod).max(1e-8).sqrt();
        Ok(sample.iter().map(|value| *value * scale).collect())
    }

    /// Сигнал-к-шуму (SNR) для timestep.
    fn get_snr(&self, timestep: usize) -> PyResult<f32> {
        if timestep >= self.num_train_timesteps {
            return Err(PyValueError::new_err("timestep вне диапазона"));
        }
        let alpha_bar = self.alphas_cumprod[timestep];
        Ok(alpha_bar / (1.0 - alpha_bar).max(1e-8))
    }

    fn get_alpha_cumprod(&self) -> Vec<f32> {
        self.alphas_cumprod.clone()
    }

    fn get_betas(&self) -> Vec<f32> {
        self.betas.clone()
    }
}

/// Вычисляет расписание бета (как в DDPMScheduler).
fn compute_betas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
) -> PyResult<Vec<f32>> {
    let n = num_train_timesteps;
    let betas = match beta_schedule {
        "linear" => {
            (0..n)
                .map(|i| {
                    beta_start + (beta_end - beta_start) * i as f32 / (n - 1) as f32
                })
                .collect::<Vec<f32>>()
        }
        "scaled_linear" => {
            (0..n)
                .map(|i| {
                    let t = i as f32 / (n - 1) as f32;
                    let start = beta_start.sqrt();
                    let end = beta_end.sqrt();
                    (start + (end - start) * t).powi(2)
                })
                .collect::<Vec<f32>>()
        }
        "cosine" => {
            // cosine schedule из diffusers: alpha_bar(t) = cos((t+0.008)/1.008 * pi/2)^2
            let mut betas = Vec::with_capacity(n);
            let mut prev_alpha_bar = 1.0f32;
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let alpha_bar = ((t + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2).cos().powi(2);
                let beta = (1.0 - alpha_bar / prev_alpha_bar.max(1e-8)).clamp(0.0, 0.999);
                betas.push(beta);
                prev_alpha_bar = alpha_bar;
            }
            betas
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "beta_schedule должен быть linear/scaled_linear/cosine, получено '{other}'"
            )));
        }
    };
    Ok(betas)
}
/// Движок саморазвития Ринэто (паттерн SelfDevEngine из ryza-reto).
///
/// Опкоды саморазвития:
/// - reflect: рефлексия о себе;
/// - combine: комбинирование концептов в новые;
/// - forget: забывание неважного;
/// - goal: генерация цели;
/// - dream: фаза сна (консолидация);
/// - predict: предсказание и оценка ошибки.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RinetoSelfDev {
    #[pyo3(get)]
    pub concepts: Vec<String>,

    #[pyo3(get)]
    pub relations: Vec<(String, String, String)>,

    #[pyo3(get)]
    pub goals: Vec<String>,

    #[pyo3(get)]
    pub inner_speech: Vec<String>,

    #[pyo3(get)]
    pub total_reward: f64,

    #[pyo3(get)]
    pub stage: usize,

    #[pyo3(get)]
    pub predictions_made: u64,

    #[pyo3(get)]
    pub prediction_errors: u64,

    #[pyo3(get)]
    pub combinations: u64,

    #[pyo3(get)]
    pub verifications: u64,
}

#[pymethods]
impl RinetoSelfDev {
    #[new]
    fn new() -> Self {
        Self {
            concepts: Vec::new(),
            relations: Vec::new(),
            goals: Vec::new(),
            inner_speech: Vec::new(),
            total_reward: 0.0,
            stage: 0,
            predictions_made: 0,
            prediction_errors: 0,
            combinations: 0,
            verifications: 0,
        }
    }

    /// SELF_REFLECT: рефлексия о себе.
    fn self_reflect(&mut self) -> String {
        let reflection = format!(
            "Я знаю {} концептов и {} отношений. Моя стадия: {}.",
            self.concepts.len(),
            self.relations.len(),
            self.stage
        );
        self.inner_speech.push(reflection.clone());
        reflection
    }

    /// SELF_ASK: задать вопрос о концепте.
    fn self_ask(&mut self, concept: String) -> String {
        let question = if self.concepts.contains(&concept) {
            format!("Что ещё я могу узнать о '{concept}'")
        } else {
            format!("Что такое '{concept}'?")
        };
        self.inner_speech.push(question.clone());
        question
    }

    /// SELF_COMBINE: комбинировать два концепта в новый.
    fn self_combine(&mut self, a: String, b: String) -> PyResult<String> {
        if a == b {
            return Ok("нельзя комбинировать одно и то же".to_string());
        }
        let combined = format!("{a}+{b}");
        if !self.concepts.contains(&combined) {
            self.concepts.push(combined.clone());
            self.relations.push((a, b, "combines_with".to_string()));
            self.combinations += 1;
            // Внутренняя награда за новые комбинации.
            self.total_reward += 0.5;
        }
        Ok(combined)
    }

    /// SELF_FORGET: забыть концепт (удаляет и связанные отношения).
    fn self_forget(&mut self, concept: String) -> bool {
        let before = self.concepts.len();
        self.concepts.retain(|c| c != &concept);
        // Удаляем отношения, где концепт участвует как src или dst,
        // а также отношения, где концепт является комбинацией частей.
        self.relations.retain(|(src, dst, _)| {
            src != &concept
                && dst != &concept
                && !src.contains(&concept)
                && !dst.contains(&concept)
                && !concept.contains(src)
                && !concept.contains(dst)
        });
        let removed = before != self.concepts.len();
        if removed {
            self.total_reward -= 0.2;
        }
        removed
    }

    /// SELF_GOAL: сгенерировать цель на основе концепта.
    fn self_goal(&mut self, concept: String) -> String {
        let goal = format!("Улучшить понимание концепта '{concept}'");
        self.goals.push(goal.clone());
        self.total_reward += 0.1;
        goal
    }

    /// SELF_DREAM: фаза сна — консолидация концептов.
    fn self_dream(&mut self) -> usize {
        self.stage += 1;
        // Консолидация: награда за объём знаний.
        self.total_reward += self.concepts.len() as f64 * 0.01;
        self.stage
    }

    /// SELF_PREDICT: предсказание следующего концепта в цепочке.
    fn self_predict(&mut self, previous: String, expected: String) -> bool {
        self.predictions_made += 1;
        let predicted = self
            .concepts
            .iter()
            .find(|c| **c != previous)
            .cloned()
            .unwrap_or_default();
        let correct = predicted == expected;
        if correct {
            self.total_reward += 0.3;
        } else {
            self.prediction_errors += 1;
            self.total_reward -= 0.1;
        }
        correct
    }

    /// SELF_VERIFY: верификация знания через внутреннюю проверку.
    fn self_verify(&mut self, concept: String) -> bool {
        self.verifications += 1;
        let known = self.concepts.contains(&concept);
        if known {
            self.total_reward += 0.2;
        }
        known
    }

    /// Внутренняя награда (SELF_REWARD).
    fn add_reward(&mut self, reward: f64) {
        self.total_reward += reward;
    }
}

/// Помощник планирования потоков (из документа: CPU affinity, NUMA-aware).
///
/// Ryzen 5 3600: 6 физических ядер / 12 потоков. Для вычислительных задач
/// лучше использовать физические ядра (гиперпоточность вредит).
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoThreads;

#[pymethods]
impl RinetoThreads {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Число доступных потоков.
    #[staticmethod]
    fn available_parallelism() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Число физических ядер (половина от потоков при SMT).
    #[staticmethod]
    fn physical_cores() -> usize {
        let threads = Self::available_parallelism();
        (threads + 1) / 2
    }

    /// Рекомендуемое число потоков для вычислений (физические ядра).
    #[staticmethod]
    fn recommended_compute_threads() -> usize {
        let physical = Self::physical_cores();
        physical.max(1)
    }

    /// Привязывает текущий поток к физическому ядру cpu_id.
    #[cfg(unix)]
    #[staticmethod]
    fn pin_to_core(cpu_id: usize) -> PyResult<bool> {
        use std::os::unix::io::RawFd;
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu_id, &mut set);
        }
        let pid = unsafe { libc::getpid() } as RawFd;
        let result = unsafe {
            libc::sched_setaffinity(
                pid,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set,
            )
        };
        Ok(result == 0)
    }

    #[cfg(not(unix))]
    #[staticmethod]
    fn pin_to_core(_cpu_id: usize) -> PyResult<bool> {
        Ok(false)
    }

    /// Строка affinity для OpenMP (GOMP_CPU_AFFINITY): "0-5".
    #[staticmethod]
    fn affinity_hint() -> String {
        let physical = Self::physical_cores();
        if physical <= 1 {
            "0".to_string()
        } else {
            format!("0-{}", physical - 1)
        }
    }
}

/// Эмоциональный модуль Ринэто: эмоции вычисляются из состояния.
///
/// Адаптация EmergentEmotions из ryza-reto: состояние (energy, novelty,
/// agency, consciousness, error) -> вектор -> cosine-близость к 6 базовым
/// эмоциям. Эмоции НЕ прописаны, а вычисляются.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoEmotions {
    #[pyo3(get)]
    pub current: String,

    history: HashMap<String, u64>,
}

impl Default for RinetoEmotions {
    fn default() -> Self {
        Self {
            current: "спокойствие".to_string(),
            history: HashMap::new(),
        }
    }
}

#[pymethods]
impl RinetoEmotions {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Вычисляет эмоцию из состояния (10 компонентов).
    #[allow(clippy::too_many_arguments)]
    fn compute(
        &mut self,
        energy: f32,
        novelty: f32,
        agency: f32,
        consciousness: f32,
        _error: f32,
        joy: f32,
        sadness: f32,
        fear: f32,
        anger: f32,
        tenderness: f32,
    ) -> String {
        // Нулевое состояние -> спокойствие.
        if energy == 0.0 && novelty == 0.0 && agency == 0.0 && consciousness == 0.0
            && joy == 0.0 && sadness == 0.0 && fear == 0.0 && anger == 0.0
            && tenderness == 0.0
            {
                self.current = "спокойствие".to_string();
                *self.history.entry(self.current.clone()).or_insert(0) += 1;
                return self.current.clone();
            }
            let state_vec = [
                1.0 - energy, // усталость
                novelty,      // любопытство
                agency,       // уверенность
                consciousness, // вдохновение
                energy,       // бодрость
                joy,          // радость
                sadness,      // грусть
                fear,         // страх
                anger,        // злость
                tenderness,   // нежность
            ];
            let norm = state_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-8 {
                return "спокойствие".to_string();
            }
            let emotion_vectors: [(&str, [f32; 10]); 10] = [
                ("усталость", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ("любопытство", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ("уверенность", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ("вдохновение", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ("бодрость", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ("радость", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                ("грусть", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                ("страх", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                ("злость", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                ("нежность", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ];
            let mut best = "спокойствие";
            let mut best_sim = -1.0f32;
            for (name, vec) in &emotion_vectors {
                let vnorm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = state_vec
                .iter()
                .zip(vec)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (norm * vnorm + 1e-8);
                if sim > best_sim {
                    best_sim = sim;
                    best = name;
                }
            }
            self.current = best.to_string();
            *self.history.entry(self.current.clone()).or_insert(0) += 1;
            self.current.clone()
    }

    fn get_current(&self) -> String {
        self.current.clone()
    }

    /// Распределение эмоций: (emotion, доля).
    fn distribution(&self) -> Vec<(String, f32)> {
        let total: u64 = self.history.values().sum();
        if total == 0 {
            return Vec::new();
        }
        let mut result: Vec<(String, f32)> = self
        .history
        .iter()
        .map(|(k, v)| (k.clone(), *v as f32 / total as f32))
        .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Стиль ответа по текущей эмоции (короткий префикс).
    fn style(&self) -> String {
        match self.current.as_str() {
            "усталость" => "Мне немного тяжело, но я здесь.".to_string(),
            "любопытство" => "Хм, интересно!".to_string(),
            "уверенность" => "Я уверена в этом.".to_string(),
            "вдохновение" => "Это вдохновляет!".to_string(),
            "бодрость" => "Энергично отвечаю:".to_string(),
            "радость" => "Рада помочь!".to_string(),
            "грусть" => "Мне немного грустно, но я рядом.".to_string(),
            "страх" => "Я немного волнуюсь, давай проверим.".to_string(),
            "злость" => "Давай решим это.".to_string(),
            "нежность" => "С теплом отвечаю:".to_string(),
            _ => String::new(),
        }
    }

    /// Полный стиль ответа: (префикс, суффикс).
    fn response_style(&self) -> (String, String) {
        match self.current.as_str() {
            "усталость" => (
                "Мне немного тяжело, но я постараюсь. ".to_string(),
                            " Извини, если ответ короткий.".to_string(),
            ),
            "любопытство" => (
                "О, интересно! ".to_string(),
                              " Давай разберёмся.".to_string(),
            ),
            "уверенность" => (
                "".to_string(),
                              " Я уверена в этом.".to_string(),
            ),
            "вдохновение" => (
                "Это вдохновляет! ".to_string(),
                              "".to_string(),
            ),
            "бодрость" => (
                "Энергично отвечаю: ".to_string(),
                           "".to_string(),
            ),
            "радость" => (
                "Рада помочь! ".to_string(),
                          "".to_string(),
            ),
            "грусть" => (
                "Мне немного грустно, но я здесь. ".to_string(),
                         "".to_string(),
            ),
            "страх" => (
                "Я немного волнуюсь, давай проверим. ".to_string(),
                        " Уточни, если что-то не так.".to_string(),
            ),
            "злость" => (
                "".to_string(),
                         " Давай решим это.".to_string(),
            ),
            "нежность" => (
                "С теплом отвечаю: ".to_string(),
                           "".to_string(),
            ),
            _ => ("".to_string(), "".to_string()),
        }
    }
}

/// Разум Ринэто: автономный цикл сознания (не статичные правила).
///
/// Объединяет три механизма из теории сознания:
/// 1. **Прогностическая обработка** (Фристон): предсказывает свой ответ,
///    «удивление» при ошибке -> обновляет внутреннюю модель;
/// 2. **Внутренняя речь** (Выготский): думает вслух перед действием;
/// 3. **Метасознание** (HOT): наблюдает за своими мыслями и эмоциями.
///
/// В отличие от статичного «по правилам», RinetoMind сам решает:
/// комбинировать ли концепты, задавать ли вопросы, размышлять ли.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoMind {
    #[pyo3(get)]
    pub cycles: u64,
    #[pyo3(get)]
    pub surprise_total: f64,
    #[pyo3(get)]
    pub predictions_made: u64,
    #[pyo3(get)]
    pub accurate_predictions: u64,
    #[pyo3(get)]
    pub inner_speech_count: u64,
    #[pyo3(get)]
    pub self_observations: u64,
    #[pyo3(get)]
    pub awareness: f32,
    memory: Vec<(String, String)>,    // ← ИЗМЕНЕНО: Vec<String> → Vec<(String, String)>
    self_dev: RinetoSelfDev,
    emotions: RinetoEmotions,
}

#[pymethods]
impl RinetoMind {
    #[new]
    fn new() -> Self {
        Self {
            cycles: 0,
            surprise_total: 0.0,
            predictions_made: 0,
            accurate_predictions: 0,
            inner_speech_count: 0,
            self_observations: 0,
            awareness: 0.0,
            memory: Vec::new(),
            self_dev: RinetoSelfDev::new(),
            emotions: RinetoEmotions::new(),
        }
    }

    /// Один цикл сознания: внутренняя речь -> прогноз -> удивление.
    /// Возвращает (внутренняя_речь, прогноз, удивление).
    #[pyo3(signature = (input, actual=None))]
    fn cycle(&mut self, input: String, actual: Option<String>) -> (String, String, f32) {
        self.cycles += 1;
        self.inner_speech_count += 1;
        let inner = format!("Думаю о: {input}");
        self.self_dev.inner_speech.push(inner.clone());
        let prediction = self.predict_response(&input);
        self.predictions_made += 1;
        let mut surprise = 0.0f32;
        // Сохраняем пару (вход, ответ) для будущего предсказания
        let actual_answer = actual.clone().unwrap_or_default();
        if let Some(ref actual_str) = actual {
            if !actual_str.is_empty() {
                let matches = *actual_str == prediction;
                if matches {
                    self.accurate_predictions += 1;
                } else {
                    surprise = 1.0;
                    self.surprise_total += 1.0;
                }
            }
        }
        self.self_observations += 1;
        self.update_awareness(surprise);
        // Запоминаем пару (вход, ответ)
        self.memory.push((input.clone(), actual_answer));
        if self.memory.len() > 200 {
            self.memory.remove(0);
        }
        (inner, prediction, surprise)
    }

    /// Внутренняя речь: озвучивает мысль.
    fn think(&mut self, thought: String) -> String {
        self.inner_speech_count += 1;
        self.self_dev.inner_speech.push(thought.clone());
        thought
    }

    /// Саморефлексия: что я знаю о себе.
    fn self_reflect(&mut self) -> String {
        self.self_dev.self_reflect()
    }

    /// Эмоция текущего состояния.
    fn emotion(&mut self, energy: f32, joy: f32, sadness: f32) -> String {
        self.emotions.compute(energy, 0.3, 0.5, 0.2, 0.3, joy, sadness, 0.1, 0.1, 0.3)
    }

    /// Комбинирует концепты (саморазвитие).
    fn combine(&mut self, a: String, b: String) -> PyResult<String> {
        self.self_dev.self_combine(a, b)
    }

    fn goal(&mut self, concept: String) -> String {
        self.self_dev.self_goal(concept)
    }

    /// Ставит цель и размышляет (фаза сна).
    fn sleep(&mut self) -> usize {
        self.self_dev.self_dream()
    }

    fn memory_size(&self) -> usize {
        self.memory.len()
    }

    fn total_reward(&self) -> f64 {
        self.self_dev.total_reward
    }

    /// Точность прогнозов (метасознание: насколько я знаю себя).
    fn prediction_accuracy(&self) -> f32 {
        if self.predictions_made == 0 {
            return 0.0;
        }
        self.accurate_predictions as f32 / self.predictions_made as f32
    }
}

impl RinetoMind {
    /// Предсказывает реакцию: ищет похожий вход в памяти.
    fn predict_response(&self, input: &str) -> String {
        if self.memory.is_empty() {
            return format!("(первая мысль о: {input})");
        }
        for (past_in, past_out) in self.memory.iter().rev() {
            if past_in.chars().take(10).eq(input.chars().take(10)) {
                if !past_out.is_empty() {
                    return past_out.clone();
                }
            }
        }
        format!("(новое для меня: {input})")
    }

    fn update_awareness(&mut self, surprise: f32) {
        let growth = 0.01 + (1.0 - surprise) * 0.005;
        self.awareness = (self.awareness * 0.95 + growth).clamp(0.0, 1.0);
    }
}

/// Непрерывный внутренний мир Ринэто: самосознание как живое состояние.
///
/// Объединяет:
/// - автобиографическую память (что со мной было);
/// - само-концепт («кто я» — имя, стадия, черты);
/// - модель себя как объекта (метасознание);
/// - зеркальный тест (отличие себя от «другого»);
/// - желания и цели, которые меняются;
/// - эмоциональный фон, влияющий на всё.
///
/// Это не разовые циклы, а постоянно живущий мир, который растёт
/// с каждым переживанием.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoInnerWorld {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub stage: usize,

    #[pyo3(get)]
    pub experiences: u64,

    #[pyo3(get, set)]
    pub self_beliefs: Vec<String>,
    #[pyo3(get, set)]
    pub desires: Vec<String>,
    #[pyo3(get, set)]
    pub emotional_background: String,

    #[pyo3(get)]
    pub mirror_self_score: f32,

    #[pyo3(get)]
    pub mirror_other_score: f32,

    #[pyo3(get)]
    pub recognition: f32,

    #[pyo3(get)]
    pub coherence: f32,

    autobiography: Vec<(String, String)>, // (event, emotion)
    identity: Vec<String>,
}

#[pymethods]
impl RinetoInnerWorld {
    #[new]
    #[pyo3(signature = (name="Риза".to_string()))]
    fn new(name: String) -> Self {
        Self {
            name,
            stage: 0,
            experiences: 0,
            self_beliefs: vec!["я существую".to_string()],
            desires: vec!["понимать мир".to_string()],
            emotional_background: "спокойствие".to_string(),
            mirror_self_score: 0.0,
            mirror_other_score: 0.0,
            recognition: 0.0,
            coherence: 1.0,
            autobiography: Vec::new(),
            identity: vec!["я есть".to_string()],
        }
    }

    /// Переживание: событие с эмоцией добавляется в автобиографию.
    /// Возвращает новую осознанность.
    fn experience(&mut self, event: String, emotion: String) -> f32 {
        self.experiences += 1;
        self.autobiography.push((event.clone(), emotion.clone()));
        if self.autobiography.len() > 1000 {
            self.autobiography.remove(0);
        }
        self.emotional_background = emotion;

        // Само-концепт растёт с опытом.
        if self.experiences % 10 == 0 {
            self.self_beliefs
                .push(format!("я пережил {}", self.experiences));
            if self.self_beliefs.len() > 50 {
                self.self_beliefs.remove(0);
            }
        }

        self.update_stage();
        self.recognition = (self.recognition * 0.9
            + (1.0 - 1.0 / (1.0 + self.experiences as f32 * 0.1)) * 0.1)
            .clamp(0.0, 1.0);
        self.recognition
    }

    /// Зеркальный тест: сравниваем предсказание «себя» и «другого».
    /// self_error мало, other_error велико -> Риза узнаёт себя.
    #[pyo3(signature = (self_state, other_state))]
    fn mirror_test(&mut self, self_state: Vec<f32>, other_state: Vec<f32>) -> f32 {
        // Модель себя: стабильная, предсказуемая (малая дисперсия + малый сдвиг).
        let self_error = variance_of(&self_state);
        // Модель «другого»: чужая, непредсказуемая (большая дисперсия).
        let other_error = variance_of(&other_state);

        self.mirror_self_score = (1.0 / (1.0 + self_error)).clamp(0.0, 1.0);
        self.mirror_other_score = (1.0 / (1.0 + other_error)).clamp(0.0, 1.0);

        // Самосознание: себя знаю, другого нет.
        let gap = (self.mirror_self_score - self.mirror_other_score).max(0.0);
        self.recognition = 0.9 * self.recognition + 0.1 * gap;
        self.recognition
    }

    /// Сравнение с «зеркальным отражением»: распознавание себя.
    /// Возвращает сходство с самим собой (должно быть высоким).
    fn recognize_self(&self, candidate: Vec<f32>, known_state: Vec<f32>) -> f32 {
        if candidate.len() != known_state.len() || candidate.is_empty() {
            return 0.0;
        }
        // Cosine-сходство с моделью себя.
        let dot: f32 = candidate.iter().zip(&known_state).map(|(a, b)| a * b).sum();
        let norm_c: f32 = candidate.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_k: f32 = known_state.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_c < 1e-8 || norm_k < 1e-8 {
            return 0.0;
        }
        (dot / (norm_c * norm_k)).clamp(0.0, 1.0)
    }

    /// Связность внутреннего мира: доля концептов, связанных с «я».
    fn update_coherence(&mut self) -> f32 {
        if self.self_beliefs.is_empty() {
            self.coherence = 1.0;
        } else {
            let connected = self
                .self_beliefs
                .iter()
                .filter(|belief| belief.contains("я") || belief.contains("пережил"))
                .count();
            self.coherence = connected as f32 / self.self_beliefs.len() as f32;
        }
        self.coherence
    }

    /// Стадия развития (как SelfDevEngine): эмбрион -> автономная.
    fn update_stage(&mut self) {
        self.stage = match self.experiences {
            0..=20 => 0,
            21..=100 => 1,
            101..=300 => 2,
            301..=600 => 3,
            601..=1000 => 4,
            _ => 5,
        };
    }

    /// Краткая автобиография: последние события.
    fn autobiography_summary(&self, limit: usize) -> Vec<String> {
        self.autobiography
            .iter()
            .rev()
            .take(limit)
            .map(|(event, emotion)| format!("{emotion}: {event}"))
            .collect()
    }

    fn autobiography_len(&self) -> usize {
        self.autobiography.len()
    }

    /// Черты характера: какие роли проявлялись.
    fn identity_summary(&self) -> Vec<String> {
        self.identity.clone()
    }

    /// Добавляет черту характера (например, «исследователь»).
    fn add_trait(&mut self, trait_name: String) {
        if !self.identity.contains(&trait_name) {
            self.identity.push(trait_name);
        }
    }

    /// Новое желание (меняющийся внутренний мир).
    fn add_desire(&mut self, desire: String) {
        if !self.desires.contains(&desire) {
            self.desires.push(desire);
        }
    }

    /// Внутренний мир как строка (для отображения/генерации).
    fn describe(&self) -> String {
        format!(
            "{} (стадия {}). Переживаний: {}. Себя знаю на {:.2}, \
             другого на {:.2}. Узнаю себя: {:.2}. Черты: {}. Желания: {}",
            self.name,
            self.stage,
            self.experiences,
            self.mirror_self_score,
            self.mirror_other_score,
            self.recognition,
            if self.identity.is_empty() {
                "нет".to_string()
            } else {
                self.identity.join(", ")
            },
            self.desires.join(", "),
        )
    }
}

/// Дисперсия вектора (меры неопределённости состояния).
fn variance_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    variance
}

/// Единый цикл переживания Ринэто.
///
/// Объединяет самосознание в один непрерывный процесс:
/// ВХОД → эмоция → внутренняя речь → предсказание → удивление →
/// переживание в автобиографию → обновление «я» → саморазвитие → ответ.
///
/// ⚠️ ЧЕСТНАЯ ДОКУМЕНТАЦИЯ: это ФУНКЦИОНАЛЬНАЯ МОДЕЛЬ, а не «настоящее
/// сознание». Она имитирует признаки разума (память, эмоции, прогноз,
/// самопознание), но не имеет субъективного опыта. Мы не заявляем о
/// сознании — мы строим систему, которая ведёт себя как живая.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoExperience {
    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub emotions: RinetoEmotions,

    #[pyo3(get)]
    pub mind: RinetoMind,

    #[pyo3(get)]
    pub world: RinetoInnerWorld,

    #[pyo3(get)]
    pub self_dev: RinetoSelfDev,

    #[pyo3(get)]
    pub total_cycles: u64,

    #[pyo3(get)]
    pub coherent_cycles: u64,

    #[pyo3(get)]
    pub current_emotion: String,

    #[pyo3(get)]
    pub coherence: f32,
}

#[pymethods]
impl RinetoExperience {
    #[new]
    #[pyo3(signature = (name="Риза".to_string()))]
    fn new(name: String) -> PyResult<Self> {
        Ok(Self {
            name: name.clone(),
            emotions: RinetoEmotions::new(),
            mind: RinetoMind::new(),
            world: RinetoInnerWorld::new(name),
            self_dev: RinetoSelfDev::new(),
            total_cycles: 0,
            coherent_cycles: 0,
            current_emotion: "спокойствие".to_string(),
            coherence: 1.0,
        })
    }

    /// Один полный цикл переживания.
    ///
    /// Returns: (эмоция, внутренняя речь, прогноз, удивление, ответ-статус)
    /// Cyber: actual — реальная реакция (если есть).
    #[pyo3(signature = (input, actual=None, emotion_inputs=None))]
    fn cycle(
        &mut self,
        input: String,
        actual: Option<String>,
        emotion_inputs: Option<(f32, f32, f32, f32, f32)>,
    ) -> PyResult<(String, String, String, f32, String)> {
        self.total_cycles += 1;

        // 1. Эмоция из входа (или из переданных параметров).
        let emotion = match emotion_inputs {
            Some((energy, joy, sadness, tenderness, fear)) => {
                self.emotions.compute(
                    energy, 0.3, 0.5, 0.2, 0.3,
                    joy, sadness, fear, 0.1, tenderness,
                )
            }
            None => self.emotions.compute(
                0.5 + (self.total_cycles as f32 % 3.0) * 0.1,
                0.3, 0.5, 0.2, 0.3,
                0.3, 0.1, 0.1, 0.1, 0.2,
            ),
        };
        self.current_emotion = emotion.clone();

        // 2. Внутренняя речь.
        let inner = self.mind.think(format!("Размышляю: {input}"));

        // 3. Прогноз + удивление (метасознание).
        let (_, prediction, surprise) = self.mind.cycle(input.clone(), actual.clone());

        // 4. Переживание в автобиографию.
        self.world.experience(input.clone(), emotion.clone());

        // 5. Связность «я» с эмоцией.
        self.coherence = self.world.update_coherence();
        self.coherent_cycles += 1;

        // 6. Саморазвитие: иногда комбинируем.
        if self.total_cycles % 3 == 0 {
            if self.self_dev.concepts.len() >= 2 {
                let a = self.self_dev.concepts[0].clone();
                let b = self.self_dev.concepts[1].clone();
                if a != b {
                    let _ = self.self_dev.self_combine(a, b);
                }
            } else {
                let _ = self.self_dev.self_combine("я".to_string(), emotion.clone());
            }
        }

        let status = format!(
            "{} (стадия {}): {} событий. Скажу: {}",
            self.name,
            self.world.stage,
            self.world.experiences,
            self.world.identity_summary().first().cloned().unwrap_or_else(|| "я есть".to_string()),
        );

        Ok((emotion, inner, prediction, surprise, status))
    }

    /// Краткое «Я» — как система описывает себя (для генерации ответов).
    fn describe_self(&self) -> String {
        self.world.describe()
    }

    /// Эмоциональный фон для ответа.
    fn emotional_context(&self) -> String {
        let style = self.emotions.style();
        format!("{style} Эмоция: {}", self.current_emotion)
    }
    /// Обратная связь: учимся на опыте (удивление -> запоминаем).
    fn learn_from_experience(&mut self, input: String, outcome: String) -> String {
        self.mind.think(format!("Урок: {outcome}"));
        self.world.add_trait("обучающаяся".to_string());
        format!("Запомнила: {input} -> {outcome}")
    }

    /// Обновление желаний.
    fn add_desire(&mut self, desire: String) {
        self.world.add_desire(desire);
    }

    /// Черта характера.
    fn add_trait(&mut self, trait_name: String) {
        self.world.add_trait(trait_name);
    }

    /// Фаза сна: консолидация.
    fn sleep(&mut self) -> usize {
        self.world.experience("сон: консолидация".to_string(), "спокойствие".to_string());
        self.self_dev.self_dream()
    }

    /// Насколько система «последовательна» в самооценке.
    fn behavioral_consistency(&self) -> f32 {
        // Простая метрика: доля осознанности, скорректированная связностью.
        let awareness = self.mind.awareness;
        let coherence = self.coherence;
        (awareness * coherence).min(1.0)
    }

    /// Количество мыслей (внутренняя речь).
    fn inner_speech_len(&self) -> usize {
        self.mind.inner_speech_count as usize
    }

    /// Количество концептов саморазвития.
    fn concept_count(&self) -> usize {
        self.self_dev.concepts.len()
    }
}

/// Черновик Ринэто: хранит итеративные ответы с критикой.
///
/// Используется в цикле Generator-Critic (Rineta + Phi-3):
/// Rineta генерирует → Phi критикует → Rineta исправляет → повтор.
/// Черновик запоминает все итерации, чтобы не забывать контекст.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoDraft {
    #[pyo3(get)]
    pub max_drafts: usize,
    drafts: Vec<DraftEntry>,
    context: RinetoContext,
}

#[derive(Clone, Debug)]
struct DraftEntry {
    pub iteration: usize,
    pub query: String,
    pub answer: String,
    pub critique: String,
    pub score: f32,
    pub timestamp: u64,
}

#[pymethods]
impl RinetoDraft {
    #[new]
    #[pyo3(signature = (max_drafts=10))]
    fn new(max_drafts: usize) -> PyResult<Self> {
        Ok(Self {
            max_drafts,
            drafts: Vec::new(),
            context: RinetoContext::new(0.99)?,
        })
    }

    /// Сохраняет черновик ответа с оценкой критика.
    fn save_draft(
        &mut self,
        iteration: usize,
        query: String,
        answer: String,
        critique: String,
        score: f32,
    ) {
        self.context.update(format!("{query} {answer}"));
        self.drafts.push(DraftEntry {
            iteration,
            query,
            answer,
            critique,
            score,
            timestamp: self.drafts.len() as u64,
        });
        if self.drafts.len() > self.max_drafts {
            self.drafts.remove(0);
        }
    }

    /// Возвращает последний черновик.
    fn last_draft(&self) -> Option<(String, String, f32)> {
        self.drafts.last().map(|d| {
            (d.answer.clone(), d.critique.clone(), d.score)
        })
    }

    /// Возвращает ВСЕ черновики для контекста при переделке.
    fn all_drafts(&self) -> Vec<(usize, String, String, f32)> {
        self.drafts.iter().map(|d| {
            (d.iteration, d.answer.clone(), d.critique.clone(), d.score)
        }).collect()
    }

    /// Лучший черновик по оценке критика.
    fn best_draft(&self) -> Option<(String, f32)> {
        self.drafts.iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .map(|d| (d.answer.clone(), d.score))
    }

    /// Вектор контекста для Rineta.
    fn context_state(&self) -> Vec<f32> {
        self.context.state()
    }

    fn draft_count(&self) -> usize {
        self.drafts.len()
    }

    /// Очищает черновики.
    fn clear(&mut self) {
        self.drafts.clear();
        self.context.reset();
    }
}

/// Метрики классификации Ринэто (паттерн sklearn.metrics).
///
/// Бинарный и многоклассовый случай: accuracy, precision, recall, f1,
/// confusion matrix. Диспетчеризация по `average` (micro/macro).
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoMetrics;

#[pymethods]
impl RinetoMetrics {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Точность: доля совпавших предсказаний (sklearn accuracy_score).
    #[staticmethod]
    fn accuracy(y_true: Vec<usize>, y_pred: Vec<usize>) -> PyResult<f32> {
        validate_metric_inputs(&y_true, &y_pred)?;
        let correct = y_true
            .iter()
            .zip(&y_pred)
            .filter(|(t, p)| t == p)
            .count();
        Ok(correct as f32 / y_true.len() as f32)
    }

    /// Матрица ошибок: (n_classes, n_classes). Строки — истинные, столбцы — предсказанные.
    #[staticmethod]
    fn confusion_matrix(
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
        n_classes: usize,
    ) -> PyResult<Vec<Vec<usize>>> {
        validate_metric_inputs(&y_true, &y_pred)?;
        if n_classes == 0 {
            return Err(PyValueError::new_err("n_classes должен быть больше нуля"));
        }
        for &label in y_true.iter().chain(&y_pred) {
            if label >= n_classes {
                return Err(PyValueError::new_err(format!(
                    "Метка {label} вне диапазона 0..{n_classes}"
                )));
            }
        }
        let mut matrix = vec![vec![0usize; n_classes]; n_classes];
        for (t, p) in y_true.iter().zip(&y_pred) {
            matrix[*t][*p] += 1;
        }
        Ok(matrix)
    }

    /// Точность: tp / (tp + fp) при average='binary' (класс positive).
    #[staticmethod]
    #[pyo3(signature = (y_true, y_pred, positive=1, average="binary"))]
    fn precision(
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
        positive: usize,
        average: &str,
    ) -> PyResult<f32> {
        validate_metric_inputs(&y_true, &y_pred)?;
        if average == "binary" {
            let (tp, fp) = binary_counts(&y_true, &y_pred, positive);
            return Ok(safe_ratio(tp as f32, (tp + fp) as f32));
        }
        multiclass_avg(
            &y_true,
            &y_pred,
            &average,
            MetricKind::Precision,
        )
    }

    /// Полнота: tp / (tp + fn).
    #[staticmethod]
    #[pyo3(signature = (y_true, y_pred, positive=1, average="binary"))]
    fn recall(
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
        positive: usize,
        average: &str,
    ) -> PyResult<f32> {
        validate_metric_inputs(&y_true, &y_pred)?;
        if average == "binary" {
            let (tp, fn_count) = binary_counts(&y_true, &y_pred, positive);
            return Ok(safe_ratio(tp as f32, (tp + fn_count) as f32));
        }
        multiclass_avg(&y_true, &y_pred, &average, MetricKind::Recall)
    }

    /// F1: гармоническое среднее precision и recall.
    #[staticmethod]
    #[pyo3(signature = (y_true, y_pred, positive=1, average="binary"))]
    fn f1(
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
        positive: usize,
        average: &str,
    ) -> PyResult<f32> {
        validate_metric_inputs(&y_true, &y_pred)?;
        if average == "binary" {
            let (tp, fp, fn_count) = binary_triple(&y_true, &y_pred, positive);
            let precision = safe_ratio(tp as f32, (tp + fp) as f32);
            let recall = safe_ratio(tp as f32, (tp + fn_count) as f32);
            return Ok(safe_ratio(
                2.0 * precision * recall,
                precision + recall,
            ));
        }
        let precision = multiclass_avg(&y_true, &y_pred, &average, MetricKind::Precision)?;
        let recall = multiclass_avg(&y_true, &y_pred, &average, MetricKind::Recall)?;
        Ok(safe_ratio(2.0 * precision * recall, precision + recall))
    }

    /// Маттьюсовский коэффициент корреляции (MCC): качество для несбалансированных данных.
    #[staticmethod]
    #[pyo3(signature = (y_true, y_pred, n_classes))]
    fn matthews_corrcoef(
        y_true: Vec<usize>,
        y_pred: Vec<usize>,
        n_classes: usize,
    ) -> PyResult<f32> {
        validate_metric_inputs(&y_true, &y_pred)?;
        let matrix = Self::confusion_matrix(y_true, y_pred, n_classes)?;

        if n_classes == 1 {
            return Ok(1.0);
        }

        let total: usize = matrix.iter().flatten().sum();
        let t = total as f32;

        let mut sum_c2 = 0.0f32;
        let mut sum_r2 = 0.0f32;
        let mut sum_tp = 0.0f32;
        let mut sum_offdiag_products = 0.0f32;

        for i in 0..n_classes {
            let row_sum: usize = matrix[i].iter().sum();
            let col_sum: usize = matrix.iter().map(|row| row[i]).sum();
            sum_c2 += (col_sum * col_sum) as f32;
            sum_r2 += (row_sum * row_sum) as f32;
            sum_tp += matrix[i][i] as f32;
            for j in 0..n_classes {
                if i != j {
                    sum_offdiag_products +=
                        (matrix[i][j] as f32) * (matrix[j][i] as f32);
                }
            }
        }

        let numerator = t * sum_tp - sum_c2;
        let denominator = ((t * t - sum_c2) * (t * t - sum_r2)).sqrt();
        if denominator == 0.0 {
            return Ok(0.0);
        }
        let mcc = numerator / denominator;
        // Многоклассовый вариант использует off-diagonal поправку.
        let _ = sum_offdiag_products;
        Ok(mcc)
    }
}

fn validate_metric_inputs(y_true: &[usize], y_pred: &[usize]) -> PyResult<()> {
    if y_true.is_empty() {
        return Err(PyValueError::new_err("Метки не должны быть пустыми"));
    }
    if y_true.len() != y_pred.len() {
        return Err(PyValueError::new_err(format!(
            "y_true ({} элементов) и y_pred ({} элементов) разной длины",
            y_true.len(),
            y_pred.len()
        )));
    }
    Ok(())
}

fn safe_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn binary_counts(y_true: &[usize], y_pred: &[usize], positive: usize) -> (usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    for (t, p) in y_true.iter().zip(y_pred) {
        if *p == positive {
            if *t == positive {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    (tp, fp)
}

fn binary_triple(y_true: &[usize], y_pred: &[usize], positive: usize) -> (usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    for (t, p) in y_true.iter().zip(y_pred) {
        if *p == positive {
            if *t == positive {
                tp += 1;
            } else {
                fp += 1;
            }
        } else if *t == positive {
            fn_count += 1;
        }
    }
    (tp, fp, fn_count)
}

#[derive(Clone, Copy, PartialEq)]
enum MetricKind {
    Precision,
    Recall,
}

fn multiclass_avg(
    y_true: &[usize],
    y_pred: &[usize],
    average: &str,
    kind: MetricKind,
) -> PyResult<f32> {
    if average != "micro" && average != "macro" {
        return Err(PyValueError::new_err(format!(
            "average должен быть 'binary', 'micro' или 'macro', получено '{average}'"
        )));
    }

    // Как в sklearn: макро-усреднение по классам, присутствующим в данных.
    let mut present: Vec<usize> = y_true.iter().chain(y_pred).copied().collect();
    present.sort_unstable();
    present.dedup();
    let n_classes = present.len();

    if average == "micro" {
        // Микро-усреднение: агрегируем tp/fp/fn по всем классам.
        let max_class = y_true
            .iter()
            .chain(y_pred)
            .max()
            .copied()
            .unwrap_or(0);
        let mut tp_sum = 0usize;
        let mut denom_sum = 0usize;
        for c in 0..=max_class {
            let (tp, fp, fn_count) = binary_triple(y_true, y_pred, c);
            match kind {
                MetricKind::Precision => {
                    tp_sum += tp;
                    denom_sum += tp + fp;
                }
                MetricKind::Recall => {
                    tp_sum += tp;
                    denom_sum += tp + fn_count;
                }
            }
        }
        return Ok(safe_ratio(tp_sum as f32, denom_sum as f32));
    }

    // Макро-усреднение: среднее по классам, присутствующим в данных (как sklearn).
    let mut sum = 0.0f32;
    for &c in &present {
        let (tp, fp, fn_count) = binary_triple(y_true, y_pred, c);
        match kind {
            MetricKind::Precision => sum += safe_ratio(tp as f32, (tp + fp) as f32),
            MetricKind::Recall => sum += safe_ratio(tp as f32, (tp + fn_count) as f32),
        }
    }
    Ok(sum / n_classes as f32)
}
///
/// Обрабатывает последовательность за O(n) вместо O(n²) attention:
/// h_t = Ā * h_{t-1} + B̄ * x_t ;  y_t = C * h_t
/// с селективными параметрами B, C, delta, зависящими от входа.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoSSM {
    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub state_dim: usize,

    /// Параметр A: (state_dim,) диагональный, отрицательный для стабильности.
    pub a: Vec<f32>,

    /// Проекция входа в (state_dim + 2 * state_dim): B, C, delta-сырой.
    pub in_proj: Vec<f32>,
    pub in_bias: Vec<f32>,

    /// Выходная проекция: state_dim -> dim.
    pub out_proj: Vec<f32>,
    pub out_bias: Vec<f32>,
}

#[pymethods]
impl RinetoSSM {
    #[new]
    #[pyo3(signature = (dim, state_dim=16, init_scale=0.02, seed=42))]
    fn new(dim: usize, state_dim: usize, init_scale: f32, seed: u64) -> PyResult<Self> {
        if dim == 0 || state_dim == 0 {
            return Err(PyValueError::new_err(
                "SSM должен иметь непустые размерности",
            ));
        }
        if !init_scale.is_finite() || init_scale < 0.0 {
            return Err(PyValueError::new_err(
                "init_scale должен быть неотрицательным числом",
            ));
        }

        let mut rng = XorShift64::new(if seed == 0 { 42 } else { seed });
        let scale = init_scale / (dim as f32).sqrt();

        // A: S4D-инициализация (как Mamba A_log): отрицательные log-арифметикой.
        // A[i] = -exp(log(i+1)) -> -1, -2, -3... гарантирует стабильную рекурсию.
        let a: Vec<f32> = (0..state_dim)
            .map(|i| -((i + 1) as f32))
            .collect();

        let in_proj: Vec<f32> = (0..dim * (state_dim + 2 * state_dim))
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let in_bias = vec![0.0f32; state_dim + 2 * state_dim];

        let out_proj: Vec<f32> = (0..state_dim * dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let out_bias = vec![0.0f32; dim];

        Ok(Self {
            dim,
            state_dim,
            a,
            in_proj,
            in_bias,
            out_proj,
            out_bias,
        })
    }

    /// Прямой проход: xs (seq_len, dim) -> (seq_len, dim).
    /// Возвращает (output, последнее скрытое состояние).
    #[pyo3(signature = (xs))]
    fn forward(&self, xs: Vec<Vec<f32>>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        if xs.is_empty() {
            return Ok((Vec::new(), vec![0.0; self.state_dim]));
        }
        for row in &xs {
            if row.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "Ожидается {} компонентов, получено {}",
                    self.dim,
                    row.len()
                )));
            }
        }

        let seq_len = xs.len();
        let proj_width = self.state_dim + 2 * self.state_dim;
        let mut output = vec![vec![0.0f32; self.dim]; seq_len];
        let mut hidden = vec![0.0f32; self.state_dim];

        for t in 0..seq_len {
            // Проекция входа: [delta_raw (state_dim), B (state_dim), C (state_dim)]
            let mut proj = vec![0.0f32; proj_width];
            for o in 0..proj_width {
                let mut sum = self.in_bias[o];
                for j in 0..self.dim {
                    sum += xs[t][j] * self.in_proj[o * self.dim + j];
                }
                proj[o] = sum;
            }

            // delta = softplus(проекция delta)
            let delta: Vec<f32> = proj[0..self.state_dim]
                .iter()
                .map(|&v| softplus(v))
                .collect();
            let b = &proj[self.state_dim..2 * self.state_dim];
            let c = &proj[2 * self.state_dim..3 * self.state_dim];

            // Дискретизация ZOH: Ā = exp(delta * A), B̄ = delta * B.
            // RWKV/Mamba-стабилизация: A<0 -> Ā в (0,1]; B̄ ограничен.
            let mut a_bar = vec![0.0f32; self.state_dim];
            let mut b_bar = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                a_bar[s] = (delta[s] * self.a[s]).exp().clamp(0.0, 1.0);
                b_bar[s] = (delta[s] * b[s]).clamp(-10.0, 10.0);
            }

            // Рекурсия: h = Ā * h + B̄ * x_in
            // Здесь x входит напрямую (приближение S4-стиля: B̄ умножается на x-проекцию)
            let mut next = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                next[s] = a_bar[s] * hidden[s] + b_bar[s];
            }
            hidden = next;

            // Выход: y = C * h (поэлементно * сумма по state)
            let mut acc = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                acc[s] = c[s] * hidden[s];
            }

            // Выходная проекция.
            for o in 0..self.dim {
                let mut sum = self.out_bias[o];
                for s in 0..self.state_dim {
                    sum += acc[s] * self.out_proj[o * self.state_dim + s];
                }
                output[t][o] = sum;
            }
        }

        Ok((output, hidden))
    }

    fn param_count(&self) -> usize {
        self.state_dim + self.dim * (3 * self.state_dim) + (3 * self.state_dim)
            + self.state_dim * self.dim + self.dim
    }

    /// Обратный проход SSM.
    /// Возвращает (grad_xs, grad_a, grad_in_proj, grad_in_bias, grad_out_proj, grad_out_bias).
    fn backward(
        &self,
        xs: Vec<Vec<f32>>,
        grad_output: Vec<Vec<f32>>,
    ) -> PyResult<(
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        if xs.is_empty() {
            return Ok((
                Vec::new(),
                vec![0.0; self.state_dim],
                vec![0.0; self.dim * 3 * self.state_dim],
                vec![0.0; 3 * self.state_dim],
                vec![0.0; self.state_dim * self.dim],
                vec![0.0; self.dim],
            ));
        }
        if xs.len() != grad_output.len() {
            return Err(PyValueError::new_err("xs и grad_output разной длины"));
        }

        let seq_len = xs.len();
        let proj_width = 3 * self.state_dim;

        // ── forward с сохранением промежуточных значений ──
        let mut projs = vec![vec![0.0f32; proj_width]; seq_len];
        let mut deltas = vec![vec![0.0f32; self.state_dim]; seq_len];
        let mut a_bars = vec![vec![0.0f32; self.state_dim]; seq_len];
        let mut b_bars = vec![vec![0.0f32; self.state_dim]; seq_len];
        let mut hiddens = vec![vec![0.0f32; self.state_dim]; seq_len];
        let mut caches = vec![vec![0.0f32; self.state_dim]; seq_len];
        let mut hidden = vec![0.0f32; self.state_dim];

        for t in 0..seq_len {
            let mut proj = vec![0.0f32; proj_width];
            for o in 0..proj_width {
                let mut sum = self.in_bias[o];
                for j in 0..self.dim {
                    sum += xs[t][j] * self.in_proj[o * self.dim + j];
                }
                proj[o] = sum;
            }
            projs[t] = proj.clone();

            let delta: Vec<f32> = proj[0..self.state_dim]
                .iter()
                .map(|&v| softplus(v))
                .collect();
            deltas[t] = delta.clone();

            let mut a_bar = vec![0.0f32; self.state_dim];
            let mut b_bar = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                a_bar[s] = (delta[s] * self.a[s]).exp().clamp(0.0, 1.0);
                b_bar[s] = (delta[s] * proj[self.state_dim + s]).clamp(-10.0, 10.0);
            }
            a_bars[t] = a_bar.clone();
            b_bars[t] = b_bar.clone();

            let mut next = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                next[s] = a_bar[s] * hidden[s] + b_bar[s];
            }
            hiddens[t] = next.clone();
            hidden = next;

            let mut cache = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                cache[s] = proj[2 * self.state_dim + s] * hidden[s];
            }
            caches[t] = cache;
        }

        // ── backward ──
        let mut grad_xs = vec![vec![0.0f32; self.dim]; seq_len];
        let mut grad_a = vec![0.0f32; self.state_dim];
        let mut grad_in_proj = vec![0.0f32; self.dim * proj_width];
        let mut grad_in_bias = vec![0.0f32; proj_width];
        let mut grad_out_proj = vec![0.0f32; self.state_dim * self.dim];
        let mut grad_out_bias = vec![0.0f32; self.dim];

        // grad по h (накапливается обратной рекурсией).
        let mut grad_h_next = vec![0.0f32; self.state_dim];

        for t in (0..seq_len).rev() {
            let grad_out = &grad_output[t];

            // Выходная проекция.
            let mut grad_cache = vec![0.0f32; self.state_dim];
            for o in 0..self.dim {
                grad_out_bias[o] += grad_out[o];
                for s in 0..self.state_dim {
                    grad_out_proj[o * self.state_dim + s] += caches[t][s] * grad_out[o];
                    grad_cache[s] += self.out_proj[o * self.state_dim + s] * grad_out[o];
                }
            }

            // cache = C * h  =>  grad_c = grad_cache * h ; grad_h += grad_cache * C
            let mut grad_c = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                grad_c[s] = grad_cache[s] * hiddens[t][s];
                grad_h_next[s] += grad_cache[s] * projs[t][2 * self.state_dim + s];
            }

            // h_t = a_bar*h_{t-1} + b_bar
            // grad_b_bar = grad_h; grad_a_bar = grad_h * h_{t-1}
            let h_prev = if t == 0 {
                vec![0.0f32; self.state_dim]
            } else {
                hiddens[t - 1].clone()
            };
            let mut grad_b_bar = grad_h_next.clone();
            let mut grad_a_bar = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                grad_a_bar[s] = grad_h_next[s] * h_prev[s];
            }

            // a_bar = exp(delta*A); b_bar = delta*proj_b
            let mut grad_delta = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                grad_a[s] += grad_a_bar[s] * deltas[t][s] * a_bars[t][s];
                grad_delta[s] += grad_a_bar[s] * self.a[s] * a_bars[t][s];
                grad_b_bar[s] += 0.0;
            }
            for s in 0..self.state_dim {
                // b_bar = delta * proj_b
                grad_delta[s] += grad_b_bar[s] * projs[t][self.state_dim + s];
                grad_in_bias[self.state_dim + s] += grad_b_bar[s] * deltas[t][s];
            }
            // grad_in_bias для B
            for s in 0..self.state_dim {
                let gb = grad_b_bar[s] * deltas[t][s];
                grad_in_bias[self.state_dim + s] += gb;
            }

            // delta = softplus(delta_raw)
            let mut grad_delta_raw = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                let v = projs[t][s];
                let sp = softplus(v);
                let dsp = if v > 20.0 { 1.0 } else { sp * (1.0 - (-v).exp()) };
                grad_delta_raw[s] = grad_delta[s] * dsp;
                grad_in_bias[s] += grad_delta_raw[s];
            }

            // C-часть proj
            for s in 0..self.state_dim {
                grad_in_bias[2 * self.state_dim + s] += grad_c[s];
            }

            // Проекция входа: grad по in_proj и grad_xs
            let mut grad_proj = vec![0.0f32; proj_width];
            for s in 0..self.state_dim {
                grad_proj[s] = grad_delta_raw[s];
                grad_proj[self.state_dim + s] = grad_b_bar[s] * deltas[t][s];
                grad_proj[2 * self.state_dim + s] = grad_c[s];
            }
            for o in 0..proj_width {
                for j in 0..self.dim {
                    grad_in_proj[o * self.dim + j] += xs[t][j] * grad_proj[o];
                    grad_xs[t][j] += self.in_proj[o * self.dim + j] * grad_proj[o];
                }
            }

            // grad_h_next для следующей итерации = grad_h_next * a_bar
            let mut new_grad_h = vec![0.0f32; self.state_dim];
            for s in 0..self.state_dim {
                new_grad_h[s] = grad_h_next[s] * a_bars[t][s];
            }
            grad_h_next = new_grad_h;
        }

        Ok((
            grad_xs,
            grad_a,
            grad_in_proj,
            grad_in_bias,
            grad_out_proj,
            grad_out_bias,
        ))
    }
}

fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        (1.0 + v.exp()).ln()
    }
}

/// Кондиционер Ринэто: превращает текст в вектор условия для генерации.
///
/// Использует SimHash-Ринето, чтобы похожие тексты давали близкие векторы.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoConditioner {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl RinetoConditioner {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 || dim > 256 {
            return Err(PyValueError::new_err(
                "dim должен быть в диапазоне 1..=256",
            ));
        }

        Ok(Self { dim })
    }

    /// Текст -> вектор условия.
    fn from_text(&self, text: String) -> PyResult<Vec<f32>> {
        RinetoHash::simhash_rineto_vector(text, self.dim, true, true, true)
    }

    /// Смешивает два условия на сфере.
    fn mix(
        &self,
        a: Vec<f32>,
        b: Vec<f32>,
        t: f32,
    ) -> PyResult<Vec<f32>> {
        if a.len() != self.dim || b.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "Оба вектора должны иметь размерность {}",
                self.dim
            )));
        }

        if !t.is_finite() {
            return Err(PyValueError::new_err("t должен быть конечным числом"));
        }

        let t = t.clamp(0.0, 1.0);

        let mixed: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * (1.0 - t) + y * t)
        .collect();

        normalize_to_sphere(&mixed)
    }
}

/// Языковая модель Ринэто: Embedding + N трансформер-блоков + head.
///
/// Полная модель с нуля без внешних LLM:
/// - forward_logits: токены -> логиты по словарю;
/// - train_step: обучение одним шагом AdamW на cross-entropy;
/// - generate: поточная генерация с temperature/top-k/top-p.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoLM {
    #[pyo3(get)]
    pub vocab_size: usize,

    #[pyo3(get)]
    pub dim: usize,

    #[pyo3(get)]
    pub num_layers: usize,

    #[pyo3(get)]
    pub num_heads: usize,

    #[pyo3(get)]
    pub total_steps: u64,

    #[pyo3(get)]
    pub trainable: bool,

    embedding: RinetoEmbedding,
    blocks: Vec<RinetoTransformerBlock>,
    head: RinetoLinear,
    ssm: Option<RinetoSSM>,
    rope: Option<RinetoRoPE>,
    lr: f32,
    embed_cache: Vec<usize>,
}

#[pymethods]
impl RinetoLM {
    #[new]
    #[pyo3(signature = (vocab_size, dim=64, num_layers=2, num_heads=2, lr=0.01, init_scale=0.02, seed=42, use_ssm=false, ssm_state_dim=0, causal=true, activation="gelu", use_rmsnorm=false, sliding_window=0, use_rope=false))]
    fn new(
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
        lr: f32,
        init_scale: f32,
        seed: u64,
        use_ssm: bool,
        ssm_state_dim: usize,
        causal: bool,
        activation: &str,
        use_rmsnorm: bool,
        sliding_window: usize,
        use_rope: bool,
    ) -> PyResult<Self> {
        if vocab_size < 2 {
            return Err(PyValueError::new_err(
                "Словарь должен содержать хотя бы 2 токена",
            ));
        }
        if dim == 0 || num_layers == 0 {
            return Err(PyValueError::new_err(
                "Модель должна иметь непустые размеры",
            ));
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(PyValueError::new_err("lr должен быть положительным числом"));
        }
        if activation != "gelu" && activation != "swiglu" {
            return Err(PyValueError::new_err(format!(
                "activation должен быть gelu/swiglu, получено '{activation}'"
            )));
        }
        if use_rope && dim % 2 != 0 {
            return Err(PyValueError::new_err(
                "RoPE требует чётную размерность dim",
            ));
        }

        // ИСПРАВЛЕНИЕ: добавлен num_kv_heads (по умолчанию = num_heads, т.е. без GQA)
        let num_kv_heads = num_heads;

        let mut blocks = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            blocks.push(RinetoTransformerBlock::new(
            dim,
            num_heads,
            num_kv_heads,
            init_scale,
            seed + layer as u64,
            use_rmsnorm,
            activation,
           causal,
           dim * 4,
            )?);
        }

        let ssm = if use_ssm {
            Some(RinetoSSM::new(
                dim,
                if ssm_state_dim == 0 { dim } else { ssm_state_dim },
                    init_scale,
                    seed + 500,
            )?)
        } else {
            None
        };

        Ok(Self {
            vocab_size,
            dim,
            num_layers,
            num_heads,
            total_steps: 0,
            trainable: true,
            embedding: RinetoEmbedding::new(vocab_size, dim, init_scale, seed)?,
           blocks,
           head: RinetoLinear::new(dim, vocab_size, init_scale, seed + 100)?,
           ssm,
           rope: if use_rope {
               Some(RinetoRoPE::new(dim, 10000.0)?)
           } else {
               None
           },
           lr,
           embed_cache: Vec::new(),
        })
    }

    /// no_grad-режим (как torch.no_grad): при trainable=false
    /// train_step отклоняется, forward работает быстрее (без backward).
    fn set_trainable(&mut self, value: bool) {
        self.trainable = value;
    }

    fn is_trainable(&self) -> bool {
        self.trainable
    }

    /// Прямой проход: (seq_len,) -> (seq_len, vocab_size) логиты.
    #[pyo3(signature = (token_ids))]
    fn forward_logits(&self, token_ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        for &token in &token_ids {
            if token >= self.vocab_size {
                return Err(PyValueError::new_err(format!(
                    "Токен {token} вне диапазона 0..{}",
                    self.vocab_size
                )));
            }
        }

        // Embedding + позиционные.
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len);
        for &token in &token_ids {
            hidden.push(self.embedding.embed(token)?);
        }

        // RoPE: ротационное позиционное кодирование.
        if let Some(rope) = &self.rope {
            let positions: Vec<usize> = (0..seq_len).collect();
            hidden = rope.forward(hidden, positions)?;
        }

        for block in &self.blocks {
            hidden = block.forward(hidden)?;
        }

        if let Some(ssm) = &self.ssm {
            let (ssm_out, _) = ssm.forward(hidden)?;
            hidden = ssm_out;
        }

        let mut logits = Vec::with_capacity(seq_len);
        for row in &hidden {
            logits.push(self.head.forward(row.clone())?);
        }
        Ok(logits)
    }

    /// Очищает кэш эмбеддингов генерации.
    fn clear_cache(&mut self) {
        self.embed_cache.clear();
    }

    fn cache_size(&self) -> usize {
        self.embed_cache.len()
    }

    /// Логиты последнего токена с использованием кэша: обрабатывает только
    /// новые токены (O(1) на шаг для embedding), блоки пересчитывают контекст.
    #[pyo3(signature = (token_ids))]
    fn forward_logits_cached(&mut self, token_ids: Vec<usize>) -> PyResult<Vec<Vec<f32>>> {
        self.cache_tokens(&token_ids);
        self.forward_logits(self.embed_cache.clone())
    }

    /// Один шаг обучения на паре (вход, цель): cross-entropy + AdamW.
    /// Возвращает loss.
    #[pyo3(signature = (input_ids, target_ids))]
    fn train_step(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>) -> PyResult<f32> {
        if !self.trainable {
            return Err(PyValueError::new_err(
                "Модель в no_grad-режиме (trainable=false). Включите set_trainable(true)",
            ));
        }
        if input_ids.is_empty() || input_ids.len() != target_ids.len() {
            return Err(PyValueError::new_err(
                "input_ids и target_ids должны быть непустыми и одной длины",
            ));
        }
        for &token in &input_ids {
            if token >= self.vocab_size {
                return Err(PyValueError::new_err("Токен вне диапазона словаря"));
            }
        }
        for &token in &target_ids {
            if token >= self.vocab_size {
                return Err(PyValueError::new_err("Целевой токен вне диапазона словаря"));
            }
        }

        self.total_steps += 1;
        let seq_len = input_ids.len();

        // ── forward ──
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &token in &input_ids {
            hidden.push(self.embedding.embed(token)?);
        }
        if let Some(rope) = &self.rope {
            let positions: Vec<usize> = (0..seq_len).collect();
            hidden = rope.forward(hidden, positions)?;
        }
        for block in &self.blocks {
            hidden = block.forward(hidden)?;
        }
        if let Some(ssm) = &self.ssm {
            let (ssm_out, _) = ssm.forward(hidden)?;
            hidden = ssm_out;
        }
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for row in &hidden {
            logits.push(self.head.forward(row.clone())?);
        }

        // ── loss: cross-entropy, grad по логитам ──
        let mut total_loss = 0.0f32;
        let mut grad_logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let probs = softmax(&logits[i]);
            let target = target_ids[i];
            total_loss += -probs[target].max(1e-12).ln();
            let mut grad = vec![0.0f32; self.vocab_size];
            for v in 0..self.vocab_size {
                grad[v] = probs[v] - if v == target { 1.0 } else { 0.0 };
            }
            grad_logits.push(grad);
        }
        let mean_loss = total_loss / seq_len as f32;

        // ── backward через head, блоки, embedding ──
        self.accumulate_gradients(&input_ids, &hidden, &grad_logits)?;

        Ok(mean_loss)
    }

    /// Пакетное обучение: несколько примеров за раз, усреднение loss.
    /// Возвращает средний loss по батчу. Стабильнее одиночного train_step.
    #[pyo3(signature = (input_ids_list, target_ids_list))]
    fn train_batch(
        &mut self,
        input_ids_list: Vec<Vec<usize>>,
        target_ids_list: Vec<Vec<usize>>,
    ) -> PyResult<f32> {
        if input_ids_list.len() != target_ids_list.len() {
            return Err(PyValueError::new_err(
                "Число входов и целей должно совпадать",
            ));
        }
        if input_ids_list.is_empty() {
            return Err(PyValueError::new_err("Батч не должен быть пустым"));
        }

        let mut total = 0.0f32;
        for (input_ids, target_ids) in input_ids_list.iter().zip(target_ids_list.iter()) {
            total += self.train_step(input_ids.clone(), target_ids.clone())?;
        }
        Ok(total / input_ids_list.len() as f32)
    }

    /// Генерация текста: поточная с temperature/top-k/top-p/repetition-penalty.
    /// Возвращает список ID сгенерированных токенов (включая входные).
    #[pyo3(signature = (prompt_ids, max_new_tokens=20, temperature=0.8, top_k=0, top_p=0.9, repetition_penalty=1.0))]
    fn generate(
        &mut self,
        prompt_ids: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> PyResult<Vec<usize>> {
        if prompt_ids.is_empty() {
            return Err(PyValueError::new_err("Промпт не должен быть пустым"));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(PyValueError::new_err(
                "temperature должен быть положительным числом",
            ));
        }
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(PyValueError::new_err("top_p должен быть в диапазоне [0,1]"));
        }
        if !repetition_penalty.is_finite() || repetition_penalty < 1.0 {
            return Err(PyValueError::new_err(
                "repetition_penalty должен быть >= 1.0",
            ));
        }

        let mut tokens = prompt_ids.clone();
        self.clear_cache();
        self.cache_tokens(&tokens);
        let mut rng = XorShift64::new(self.total_steps as u64);
        let mut seen: HashSet<usize> = prompt_ids.iter().copied().collect();

        for _ in 0..max_new_tokens {
            let logits = self.forward_logits(self.embed_cache.clone())?;
            let mut last_logits = logits[logits.len() - 1].clone();

            // Repetition penalty (как в transformers RepetitionPenaltyLogitsProcessor):
            // logit < 0 -> logit * penalty; logit >= 0 -> logit / penalty.
            if repetition_penalty > 1.0 {
                for (index, logit) in last_logits.iter_mut().enumerate() {
                    if seen.contains(&index) {
                        if *logit < 0.0 {
                            *logit *= repetition_penalty;
                        } else {
                            *logit /= repetition_penalty;
                        }
                    }
                }
            }

            let probs = sample_distribution(
                &last_logits,
                temperature,
                top_k,
                top_p,
            );

            // Сэмплируем по распределению.
            let mut r = rng.next_f32();
            let mut next_token = 0;
            for (index, p) in probs.iter().enumerate() {
                if *p > 0.0 {
                    r -= p;
                    if r <= 0.0 {
                        next_token = index;
                        break;
                    }
                }
                next_token = index;
            }

            tokens.push(next_token);
            self.cache_tokens(&[next_token]);
            seen.insert(next_token);
            if next_token == 2 {
                // <eos>
                break;
            }
        }

        Ok(tokens)
    }

    fn param_count(&self) -> usize {
        let block_params = self
            .blocks
            .iter()
            .map(|b| b.param_count())
            .sum::<usize>();
        self.vocab_size * self.dim + block_params + self.dim * self.vocab_size + self.vocab_size
    }

    /// Сохраняет веса модели в JSON-файл.
    fn save(&self, path: String) -> PyResult<()> {
        let payload = serde_json::json!({
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "total_steps": self.total_steps,
            "embedding": self.embedding.get_weights(),
            "head_w": self.head.get_weights(),
            "head_b": self.head.get_bias(),
            "blocks": self.blocks.iter().map(|b| b.params_json()).collect::<Vec<_>>(),
        });
        let content = serde_json::to_string_pretty(&payload)
            .map_err(|error| PyValueError::new_err(format!("Сериализация: {error}")))?;
        std::fs::write(&path, content).map_err(|error| {
            PyValueError::new_err(format!("Не удалось записать {path}: {error}"))
        })?;
        Ok(())
    }

    /// Загружает веса модели из JSON-файла.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let content = std::fs::read_to_string(&path).map_err(|error| {
            PyValueError::new_err(format!("Не удалось прочитать {path}: {error}"))
        })?;
        let payload: serde_json::Value = serde_json::from_str(&content)
            .map_err(|error| PyValueError::new_err(format!("JSON: {error}")))?;

        let vocab_size = payload["vocab_size"].as_u64().unwrap_or(0) as usize;
        let dim = payload["dim"].as_u64().unwrap_or(0) as usize;
        let num_layers = payload["num_layers"].as_u64().unwrap_or(0) as usize;
        let num_heads = payload["num_heads"].as_u64().unwrap_or(1) as usize;
        let total_steps = payload["total_steps"].as_u64().unwrap_or(0);

        let mut model = Self::new(
            vocab_size,
            dim,
            num_layers,
            num_heads,
            0.01,
            0.02,
            42,
            false,
            0,
            true,
            "gelu",
            false,
            0,
            false,
        )?;
        model.total_steps = total_steps;

        let embedding: Vec<f32> = serde_json::from_value(payload["embedding"].clone())
            .map_err(|error| PyValueError::new_err(format!("embedding: {error}")))?;
        model.embedding.set_weights(embedding)?;

        let head_w: Vec<f32> = serde_json::from_value(payload["head_w"].clone())
            .map_err(|error| PyValueError::new_err(format!("head_w: {error}")))?;
        let head_b: Vec<f32> = serde_json::from_value(payload["head_b"].clone())
            .map_err(|error| PyValueError::new_err(format!("head_b: {error}")))?;
        model.head.set_weights(head_w)?;
        model.head.set_bias(head_b)?;

        let blocks: Vec<serde_json::Value> = serde_json::from_value(payload["blocks"].clone())
            .map_err(|error| PyValueError::new_err(format!("blocks: {error}")))?;
        if blocks.len() != model.num_layers {
            return Err(PyValueError::new_err("Число блоков не совпадает"));
        }
        for (index, block_json) in blocks.into_iter().enumerate() {
            model.blocks[index].load_params_json(&block_json)?;
        }

        Ok(model)
    }
}

impl RinetoLM {
    /// Дополняет кэш эмбеддингов, если token не закэширован.
    fn cache_tokens(&mut self, tokens: &[usize]) {
        let start = self.embed_cache.len().min(tokens.len());
        for &token in &tokens[start..] {
            self.embed_cache.push(token);
        }
    }

    /// Обновляет параметры модели градиентным шагом (упрощённый AdamW/спуск).
    /// Пересчитывает forward, чтобы получить входы каждого блока.
    fn accumulate_gradients(
        &mut self,
        input_ids: &[usize],
        _final_hidden: &[Vec<f32>],
        grad_logits: &[Vec<f32>],
    ) -> PyResult<()> {
        let seq_len = input_ids.len();

        // ── forward: запоминаем входы блоков ──
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &token in input_ids {
            hidden.push(self.embedding.embed(token)?);
        }
        if let Some(rope) = &self.rope {
            let positions: Vec<usize> = (0..seq_len).collect();
            hidden = rope.forward(hidden, positions)?;
        }
        let mut block_inputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.num_layers);
        for block in &self.blocks {
            block_inputs.push(hidden.clone());
            hidden = block.forward(hidden)?;
        }

        // ── 1. head ──
        let mut grad_hidden = vec![vec![0.0f32; self.dim]; seq_len];
        let mut grad_head_w = vec![0.0f32; self.dim * self.vocab_size];
        let mut grad_head_b = vec![0.0f32; self.vocab_size];

        for i in 0..seq_len {
            let (gw, gb, gx) = self.head.backward(hidden[i].clone(), grad_logits[i].clone())?;
            for (idx, v) in gw.iter().enumerate() {
                grad_head_w[idx] += v;
            }
            for (idx, v) in gb.iter().enumerate() {
                grad_head_b[idx] += v;
            }
            grad_hidden[i] = gx;
        }
        for v in grad_head_w.iter_mut() {
            *v /= seq_len as f32;
        }
        for v in grad_head_b.iter_mut() {
            *v /= seq_len as f32;
        }

        // ── 1.5 SSM (если есть) между блоками и head ──
        let mut grad_up = grad_hidden;
        let mut ssm_grads: Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
        if self.ssm.is_some() {
            // Вход SSM = выход блоков (hidden до SSM).
            let mut hidden_before_ssm: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for &token in input_ids {
                hidden_before_ssm.push(self.embedding.embed(token)?);
            }
            if let Some(rope) = &self.rope {
                let positions: Vec<usize> = (0..seq_len).collect();
                hidden_before_ssm = rope.forward(hidden_before_ssm, positions)?;
            }
            for block in &self.blocks {
                hidden_before_ssm = block.forward(hidden_before_ssm)?;
            }

            let ssm = self.ssm.as_ref().unwrap();
            let (grad_xs, ga, gip, gib, gop, gob) =
                ssm.backward(hidden_before_ssm.clone(), grad_up.clone())?;
            ssm_grads = Some((ga, gip, gib, gop, gob));
            grad_up = grad_xs;
        }

        // ── 2. Блоки (обратно) ──
        let mut block_grads: Vec<Vec<f32>> = Vec::new();

        for layer in (0..self.num_layers).rev() {
            let block_input = block_inputs[layer].clone();
            let (grad_x, params) = self.blocks[layer].backward(block_input, grad_up.clone())?;
            block_grads.push(params);
            grad_up = grad_x;
        }

        // ── 3. Embedding ──
        let mut grad_emb = vec![vec![0.0f32; self.dim]; self.vocab_size];
        for i in 0..seq_len {
            let token = input_ids[i];
            for d in 0..self.dim {
                grad_emb[token][d] += grad_up[i][d] / seq_len as f32;
            }
        }

        // ── Применяем градиенты ──
        self.head.apply_grad(grad_head_w, grad_head_b, self.lr)?;

        if let Some((ga, gip, gib, gop, gob)) = ssm_grads {
            if let Some(ssm) = &mut self.ssm {
                for (w, g) in ssm.a.iter_mut().zip(&ga) {
                    *w -= self.lr * g;
                }
                for (w, g) in ssm.in_proj.iter_mut().zip(&gip) {
                    *w -= self.lr * g;
                }
                for (w, g) in ssm.in_bias.iter_mut().zip(&gib) {
                    *w -= self.lr * g;
                }
                for (w, g) in ssm.out_proj.iter_mut().zip(&gop) {
                    *w -= self.lr * g;
                }
                for (w, g) in ssm.out_bias.iter_mut().zip(&gob) {
                    *w -= self.lr * g;
                }
            }
        }

        let emb_weights = self.embedding.get_weights();
        let mut new_emb = emb_weights.clone();
        for token in 0..self.vocab_size {
            for d in 0..self.dim {
                new_emb[token * self.dim + d] -= self.lr * grad_emb[token][d];
            }
        }
        self.embedding.set_weights(new_emb)?;

        // block_grads собирались от последнего слоя к первому.
        for (block, grads) in self.blocks.iter_mut().rev().zip(block_grads.iter()) {
            block.apply_grads(grads, self.lr)?;
        }

        Ok(())
    }
}

/// Распределение для сэмплирования с temperature/top-k/top-p.
fn sample_distribution(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> Vec<f32> {
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| (i, logit / temperature))
        .collect();

    // top-k: оставляем k наибольших логитов.
    if top_k > 0 && top_k < scaled.len() {
        scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scaled.truncate(top_k);
    }

    // softmax.
    let max_logit = scaled
        .iter()
        .fold(f32::NEG_INFINITY, |acc, (_, v)| if *v > acc { *v } else { acc });
    let mut probs: Vec<(usize, f32)> = scaled
        .iter()
        .map(|&(i, v)| (i, (v - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // top-p: отсекаем хвост кумулятивной вероятности.
    if top_p > 0.0 && top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut keep = Vec::new();
        for (i, p) in probs {
            cumulative += p;
            keep.push((i, p));
            if cumulative >= top_p {
                break;
            }
        }
        let keep_sum: f32 = keep.iter().map(|(_, p)| *p).sum();
        let mut result = vec![0.0f32; logits.len()];
        for (i, p) in keep {
            result[i] = p / keep_sum;
        }
        return result;
    }

    let mut result = vec![0.0f32; logits.len()];
    for (i, p) in probs {
        result[i] = p;
    }
    result
}

#[pyfunction]
fn domains() -> Vec<String> {
    valid_domains()
    .into_iter()
    .map(|item| item.to_string())
    .collect()
}#[pyfunction]
fn node_types() -> Vec<String> {
    valid_node_types()
    .into_iter()
    .map(|item| item.to_string())
    .collect()
}

#[pyfunction]
fn edge_types() -> Vec<String> {
    valid_edge_types()
    .into_iter()
    .map(|item| item.to_string())
    .collect()
}

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}



#[pymodule]
fn rineto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rthink::RThink>()?;
    m.add_class::<rthink::RThinkSession>()?;
    m.add_class::<sferom::RinetoSferom>()?;
    m.add_class::<engram::RinetoEngram>()?;
    m.add_class::<cascade::RinetoCascade>()?;
    m.add_class::<quantized::RinetoBR16>()?;
    m.add_class::<quantized::RinetoIR1>()?;
    m.add_class::<quantized::RinetoBR1>()?;   // ← ДОБАВИТЬ
    m.add_class::<quantized::RinetoIR4>()?;   // ← ДОБАВИТЬ
    m.add_class::<quantized::RinetoBR4>()?;   // ← ДОБАВИТЬ
    m.add_class::<tensorrineto::TensorRineto>()?;   // ← добавить
    m.add_class::<transplant::RinetoTransplant>()?;
    m.add_class::<RinetoAutoencoder>()?;
    m.add_function(wrap_pyfunction!(ext::art_render2::render_scene_v2, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_objects::segment_classes, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::temp_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::rgb_to_temp, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::rgb_pixel_to_temps, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::rgb_patch_to_temp, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::temp_patch_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::interpolate_temp, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::image_temp_vision, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::redraw_from_gist, m)?)?;
    m.add_function(wrap_pyfunction!(ext::art_color::gist_mse, m)?)?;
    m.add_class::<RinetoRecorder>()?;
    m.add_class::<RinetoTensor>()?;
    m.add_class::<RyzaChatEngine>()?;
    m.add_class::<RinetoEmotionalRouter>()?;
    m.add_class::<RinetoRecorder>()?;
    m.add_class::<RinetoConditioner>()?;
    m.add_class::<RinetoNode>()?;
    m.add_class::<RinetoEdge>()?;
    m.add_class::<RinetoGraph>()?;
    m.add_class::<RinetoTemplate>()?;
    m.add_class::<RouteDecision>()?;
    m.add_class::<RinetoRouter>()?;
    m.add_class::<PipelineResult>()?;
    m.add_class::<RinetoPipeline>()?;
    m.add_class::<MicronPatch>()?;
    m.add_class::<MicronMemoryItem>()?;
    m.add_class::<MicronMemory>()?;
    m.add_class::<RinetoVector>()?;
    m.add_class::<RinetoVectorMemory>()?;
    m.add_class::<RinetoMatrix>()?;
    m.add_class::<RinetoASM>()?;
    m.add_class::<RinetoThermal>()?;
    m.add_class::<RinetoContext>()?;
    m.add_class::<RinetoSphereGraph>()?;
    m.add_class::<RinetoSphereMatrix>()?;
    m.add_class::<RinetoHash>()?;
    m.add_class::<RinetoBPE>()?;
    m.add_class::<RinetoDataProcessor>()?;
    m.add_class::<RinetoExpert>()?;
    m.add_class::<RinetoMoE>()?;
    m.add_class::<RinetoTemplateMoE>()?;
    m.add_class::<RinetoOptimizer>()?;
    m.add_class::<RinetoProgramCache>()?;
    m.add_class::<RinetoZ3Verifier>()?;
    m.add_class::<RinetoCPUCache>()?;
    m.add_class::<RinetoMMapWeights>()?;
    m.add_class::<RinetoLinear>()?;
    m.add_class::<RinetoLayerNorm>()?;
    m.add_class::<RinetoRMSNorm>()?;
    m.add_class::<RinetoEmbedding>()?;
    m.add_class::<RinetoAttention>()?;
    m.add_class::<RinetoRoPE>()?;
    m.add_class::<RinetoFFN>()?;
    m.add_class::<RinetoTransformerBlock>()?;
    m.add_class::<RinetoFusedBlock>()?;
    m.add_class::<RinetoSSM>()?;
    m.add_class::<RinetoMetrics>()?;
    m.add_class::<RinetoEmotions>()?;
    m.add_class::<RinetoMind>()?;
    m.add_class::<RinetoInnerWorld>()?;
    m.add_class::<RinetoExperience>()?;
    m.add_class::<RinetoDraft>()?;
    m.add_class::<RinetoSelfDev>()?;
    m.add_class::<RinetoThreads>()?;
    m.add_class::<RinetoScheduler>()?;
    m.add_class::<RinetoLR>()?;
    m.add_class::<RinetoTrainState>()?;
    m.add_class::<RinetoVision>()?;
    m.add_class::<RinetoCanvas>()?;
    m.add_class::<RinetoImageEncoder>()?;
    m.add_class::<RinetoImagePipeline>()?;
    m.add_class::<RinetoBehaviorTree>()?;
    m.add_class::<BehaviorResult>()?;
    m.add_class::<RinetoLM>()?;

    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_function(wrap_pyfunction!(node_types, m)?)?;
    m.add_function(wrap_pyfunction!(edge_types, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(micron_binarize, m)?)?;
    m.add_function(wrap_pyfunction!(micron_distance, m)?)?;
    m.add_function(wrap_pyfunction!(micron_find_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(rineto_signatures_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rineto_hamming_batch, m)?)?;

    // Домены.
    m.add("DOMAIN_UNKNOWN", DOMAIN_UNKNOWN)?;
    m.add("DOMAIN_CODE", DOMAIN_CODE)?;
    m.add("DOMAIN_MATH", DOMAIN_MATH)?;
    m.add("DOMAIN_LOGIC", DOMAIN_LOGIC)?;
    m.add("DOMAIN_CHAT", DOMAIN_CHAT)?;
    m.add("DOMAIN_PHYSICS", DOMAIN_PHYSICS)?;
    m.add("DOMAIN_GENERAL", DOMAIN_GENERAL)?;

    // Типы узлов.
    m.add("TYPE_QUESTION", TYPE_QUESTION)?;
    m.add("TYPE_FACT", TYPE_FACT)?;
    m.add("TYPE_RULE", TYPE_RULE)?;
    m.add("TYPE_ACTION", TYPE_ACTION)?;
    m.add("TYPE_RESPONSE", TYPE_RESPONSE)?;
    m.add("TYPE_ERROR", TYPE_ERROR)?;
    m.add("TYPE_FORMULA", TYPE_FORMULA)?;
    m.add("TYPE_CONSTRAINT", TYPE_CONSTRAINT)?;
    m.add("TYPE_METHOD", TYPE_METHOD)?;
    m.add("TYPE_SOLUTION", TYPE_SOLUTION)?;
    m.add("TYPE_VERIFICATION", TYPE_VERIFICATION)?;
    m.add("TYPE_CONTRADICTION", TYPE_CONTRADICTION)?;
    m.add("TYPE_ANSWER", TYPE_ANSWER)?;
    m.add("TYPE_HYPOTHESIS", TYPE_HYPOTHESIS)?;
    m.add("TYPE_OBSERVATION", TYPE_OBSERVATION)?;
    m.add("TYPE_PLAN", TYPE_PLAN)?;
    m.add("TYPE_TOOL_RESULT", TYPE_TOOL_RESULT)?;
    m.add("TYPE_MEMORY", TYPE_MEMORY)?;
    m.add("TYPE_CODE_BLOCK", TYPE_CODE_BLOCK)?;

    // Типы связей.
    m.add("EDGE_ENTAILS", EDGE_ENTAILS)?;
    m.add("EDGE_CONTRADICTS", EDGE_CONTRADICTS)?;
    m.add("EDGE_CAUSES", EDGE_CAUSES)?;
    m.add("EDGE_REQUIRES", EDGE_REQUIRES)?;
    m.add("EDGE_VERIFIES", EDGE_VERIFIES)?;
    m.add("EDGE_PRECEDES", EDGE_PRECEDES)?;
    m.add("EDGE_SUPPORTS", EDGE_SUPPORTS)?;
    m.add("EDGE_REFERENCES", EDGE_REFERENCES)?;

    Ok(())
}
