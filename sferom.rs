use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{dot_simd, RinetoHash};

/// SFEROM 0.2
///
/// Иерархическая структура:
///
///                 CORE
///                  │
///          ┌───────┼───────┐
///          │       │       │
///        LOBE    LOBE    LOBE
///          │       │       │
///        nanos   nanos   nanos
///
/// Основные изменения относительно 0.1:
/// - centroid корректно инициализируется первым примером;
/// - centroid нормализуется;
/// - routing поддерживает TOP-K;
/// - recall ищет по всем долям, а не только в победившей;
/// - radial больше не используется как простой бонус за количество записей;
/// - score нормализован в диапазон примерно 0..1;
/// - dimension берётся из самой модели, а не жёстко 64.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RinetoSferom {
    #[pyo3(get)]
    pub num_lobes: usize,

    #[pyo3(get)]
    pub total_nanos: u64,

    lobe_names: Vec<String>,

    /// (signature, text, label)
    nanos: Vec<Vec<(u64, String, String)>>,

    /// Нормализованный центр каждой доли.
    centroids: Vec<Vec<f32>>,

    /// Сколько обучающих примеров есть в каждой доле.
    lobe_counts: Vec<u64>,

    /// Внутренний радиус сферы.
    ///
    /// Теперь это не прямой множитель score.
    radial: Vec<f32>,

    /// Глобальный центр.
    core: Vec<f32>,
}

#[pymethods]
impl RinetoSferom {
    #[new]
    #[pyo3(signature = (names, dim=64))]
    fn new(names: Vec<String>, dim: usize) -> PyResult<Self> {
        if names.is_empty() {
            return Err(PyValueError::new_err(
                "SFEROM: нужна хотя бы одна доля",
            ));
        }

        if dim == 0 {
            return Err(PyValueError::new_err(
                "SFEROM: размерность должна быть больше нуля",
            ));
        }

        let n = names.len();

        Ok(Self {
            num_lobes: n,
            total_nanos: 0,
            lobe_names: names,
            nanos: vec![Vec::new(); n],
           centroids: vec![vec![0.0; dim]; n],
           lobe_counts: vec![0; n],
           radial: vec![0.0; n],
           core: vec![0.0; dim],
        })
    }

    /// Добавляет знание в конкретную долю.
    ///
    /// Первый пример становится центром напрямую.
    /// Последующие примеры обновляют центр через EMA.
    fn learn(
        &mut self,
        lobe: usize,
        text: String,
        label: String,
    ) -> PyResult<()> {
        if lobe >= self.num_lobes {
            return Err(PyValueError::new_err(
                "SFEROM: доля вне диапазона",
            ));
        }

        if text.trim().is_empty() {
            return Err(PyValueError::new_err(
                "SFEROM: пустой текст нельзя обучить",
            ));
        }

        let dim = self.centroids[lobe].len();

        let signature = RinetoHash::simhash_rineto(
            text.clone(),
                                                   true,
                                                   true,
                                                   true,
        );

        let mut vector =
        RinetoHash::simhash_rineto_vector(
            text.clone(),
                                          dim,
                                          true,
                                          true,
                                          true,
        )?;

        normalize(&mut vector);

        self.lobe_counts[lobe] += 1;

        let count = self.lobe_counts[lobe];

        if count == 1 {
            self.centroids[lobe] = vector.clone();
        } else {
            // EMA.
            //
            // Чем больше знаний, тем меньше влияние одного нового примера.
            let alpha = if count < 10 {
                0.20
            } else {
                0.10
            };

            for i in 0..dim {
                self.centroids[lobe][i] =
                (1.0 - alpha) * self.centroids[lobe][i]
                + alpha * vector[i];
            }

            normalize(&mut self.centroids[lobe]);
        }

        // Обновляем глобальное ядро.
        if self.total_nanos == 0 {
            self.core = vector.clone();
        } else {
            let alpha = 1.0 / ((self.total_nanos + 1) as f32);

            for i in 0..dim {
                self.core[i] =
                (1.0 - alpha) * self.core[i]
                + alpha * vector[i];
            }

            normalize(&mut self.core);
        }

        self.nanos[lobe].push((
            signature,
            text,
            label,
        ));

        self.total_nanos += 1;

        // Радиус теперь зависит от количества знаний,
        // но НЕ используется как прямой бонус маршрутизации.
        self.radial[lobe] =
        (self.lobe_counts[lobe] as f32).sqrt();

        Ok(())
    }

    /// Старый API:
    ///
    /// route(text) -> (lobe_id, lobe_name, score)
    ///
    /// Возвращает лучшую долю.
    fn route(
        &self,
        text: String,
    ) -> PyResult<(usize, String, f32)> {
        let routes = self.route_topk_internal(&text, 1)?;

        if routes.is_empty() {
            return Err(PyValueError::new_err(
                "SFEROM: нет доступных долей",
            ));
        }

        let (id, score) = routes[0];

        Ok((
            id,
            self.lobe_names[id].clone(),
            score,
        ))
    }

    /// Новый API.
    ///
    /// Возвращает TOP-K долей:
    ///
    /// [
    ///   (lobe_id, name, score),
    ///   ...
    /// ]
    #[pyo3(signature = (text, k=3))]
    fn route_topk(
        &self,
        text: String,
        k: usize,
    ) -> PyResult<Vec<(usize, String, f32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let routes =
        self.route_topk_internal(&text, k)?;

        Ok(routes
        .into_iter()
        .map(|(id, score)| {
            (
                id,
             self.lobe_names[id].clone(),
             score,
            )
        })
        .collect())
    }

    /// Новый recall.
    ///
    /// ВАЖНО:
    /// раньше recall сначала выбирал одну долю,
    /// а потом искал только внутри неё.
    ///
    /// Теперь ищем по ВСЕМ сферам.
    fn recall(
        &self,
        text: String,
        k: usize,
    ) -> PyResult<Vec<(String, u32)>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let signature =
        RinetoHash::simhash_rineto(
            text,
            true,
            true,
            true,
        );

        let mut hits: Vec<(String, u32)> =
        Vec::new();

        for lobe in &self.nanos {
            for (stored_sig, _, label) in lobe {
                let distance =
                (signature ^ stored_sig).count_ones();

                hits.push((
                    label.clone(),
                           distance,
                ));
            }
        }

        hits.sort_by_key(|x| x.1);
        hits.truncate(k);

        Ok(hits)
    }

    /// Размерность SFEROM.
    fn dim(&self) -> usize {
        self.centroids
        .first()
        .map(|x| x.len())
        .unwrap_or(0)
    }

    /// Количество знаний в конкретной доле.
    fn lobe_size(&self, lobe: usize) -> PyResult<u64> {
        if lobe >= self.num_lobes {
            return Err(PyValueError::new_err(
                "SFEROM: доля вне диапазона",
            ));
        }

        Ok(self.lobe_counts[lobe])
    }

    /// Возвращает названия всех долей.
    fn lobe_names(&self) -> Vec<String> {
        self.lobe_names.clone()
    }

    /// Возвращает статистику долей.
    fn stats(&self) -> Vec<(String, u64, f32)> {
        self.lobe_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.clone(),
             self.lobe_counts[i],
             self.radial[i],
            )
        })
        .collect()
    }
}

impl RinetoSferom {
    fn route_topk_internal(
        &self,
        text: &str,
        k: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let dim = self.dim();

        if dim == 0 {
            return Err(PyValueError::new_err(
                "SFEROM: размерность не определена",
            ));
        }

        let mut query =
        RinetoHash::simhash_rineto_vector(
            text.to_string(),
                                          dim,
                                          true,
                                          true,
                                          true,
        )?;

        normalize(&mut query);

        let query_words = sig_words(text);

        let mut scores:
        Vec<(usize, f32)> = Vec::new();

        for lobe in 0..self.num_lobes {
            // Пустая доля не должна выигрывать случайно.
            if self.lobe_counts[lobe] == 0 {
                continue;
            }

            let centroid =
            &self.centroids[lobe];

            let cosine =
            dot_simd(&query, centroid)
            .clamp(-1.0, 1.0);

            // Переводим [-1, 1] -> [0, 1].
            let semantic =
            (cosine + 1.0) * 0.5;

            // Лексическое совпадение.
            let lexical = self.nanos[lobe]
            .iter()
            .map(|(_, stored, _)| {
                jaccard(
                    &query_words,
                    &sig_words(stored),
                )
            })
            .fold(0.0, f32::max);

            // Hamming similarity по nano.
            //
            // Берём лучший nano в сфере.
            let query_sig =
            RinetoHash::simhash_rineto(
                text.to_string(),
                                       true,
                                       true,
                                       true,
            );

            let best_hamming =
            self.nanos[lobe]
            .iter()
            .map(|(sig, _, _)| {
                (query_sig ^ *sig)
                .count_ones()
            })
            .min()
            .unwrap_or(64);

            let hamming_similarity =
            1.0
            - (best_hamming as f32 / 64.0);

            // Основной вес — семантика.
            //
            // Jaccard больше НЕ способен перетащить
            // совершенно другой запрос в сферу.
            let score =
            0.70 * semantic
            + 0.20 * hamming_similarity
            + 0.10 * lexical;

            scores.push((lobe, score));
        }

        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
            .unwrap_or(
                std::cmp::Ordering::Equal,
            )
        });

        scores.truncate(
            k.min(scores.len()),
        );

        Ok(scores)
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v
    .iter()
    .map(|x| x * x)
    .sum::<f32>()
    .sqrt();

    if norm <= f32::EPSILON {
        return;
    }

    for x in v {
        *x /= norm;
    }
}

fn sig_words(text: &str) -> Vec<String> {
    text.split_whitespace()
    .map(|word| {
        word.trim_matches(
            |c: char| {
                !c.is_alphanumeric()
                && c != '_'
        && c != '+'
        && c != '-'
            },
        )
        .to_lowercase()
    })
    .filter(|word| word.chars().count() > 2)
    .collect()
}

fn jaccard(
    a: &[String],
    b: &[String],
) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let inter =
    a.iter()
    .filter(|x| b.contains(x))
    .count() as f32;

    let union =
    (a.len() + b.len()) as f32
    - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}
