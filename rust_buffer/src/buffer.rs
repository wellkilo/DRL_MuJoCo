use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray;
use ndarray::{Array1, Array2, Axis};

#[pyclass]
pub struct RustReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    obs: Vec<f32>,
    actions: Vec<f32>,
    log_probs: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

#[pymethods]
impl RustReplayBuffer {
    #[new]
    pub fn new(obs_dim: usize, act_dim: usize) -> Self {
        RustReplayBuffer {
            obs_dim,
            act_dim,
            obs: Vec::new(),
            actions: Vec::new(),
            log_probs: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    pub fn add_trajectory(
        &mut self,
        obs: PyReadonlyArray2<'_, f32>,
        actions: PyReadonlyArray2<'_, f32>,
        log_probs: PyReadonlyArray1<'_, f32>,
        advantages: PyReadonlyArray1<'_, f32>,
        returns: PyReadonlyArray1<'_, f32>,
    ) {
        let obs_slice = obs.as_slice().unwrap();
        let actions_slice = actions.as_slice().unwrap();
        let log_probs_slice = log_probs.as_slice().unwrap();
        let advantages_slice = advantages.as_slice().unwrap();
        let returns_slice = returns.as_slice().unwrap();

        self.obs.extend_from_slice(obs_slice);
        self.actions.extend_from_slice(actions_slice);
        self.log_probs.extend_from_slice(log_probs_slice);
        self.advantages.extend_from_slice(advantages_slice);
        self.returns.extend_from_slice(returns_slice);
    }

    pub fn get_all_as_numpy<'py>(&self, py: Python<'py>) -> &'py PyDict {
        let num_samples = self.log_probs.len();
        
        let obs_array = Array2::from_shape_vec((num_samples, self.obs_dim), self.obs.clone()).unwrap();
        let actions_array = Array2::from_shape_vec((num_samples, self.act_dim), self.actions.clone()).unwrap();
        let log_probs_array = Array1::from_vec(self.log_probs.clone());
        let advantages_array = Array1::from_vec(self.advantages.clone());
        let returns_array = Array1::from_vec(self.returns.clone());

        let dict = PyDict::new(py);
        dict.set_item("obs", obs_array.to_pyarray(py)).unwrap();
        dict.set_item("act", actions_array.to_pyarray(py)).unwrap();
        dict.set_item("logp", log_probs_array.to_pyarray(py)).unwrap();
        dict.set_item("adv", advantages_array.to_pyarray(py)).unwrap();
        dict.set_item("ret", returns_array.to_pyarray(py)).unwrap();
        dict
    }

    pub fn normalize_advantages(&mut self) {
        if self.advantages.is_empty() {
            return;
        }

        let mean = self.advantages.iter().sum::<f32>() / self.advantages.len() as f32;
        let var = self.advantages.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.advantages.len() as f32;
        let std = var.sqrt() + 1e-8;

        for adv in &mut self.advantages {
            *adv = (*adv - mean) / std;
        }
    }

    pub fn clear(&mut self) {
        self.obs.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.advantages.clear();
        self.returns.clear();
    }

    pub fn size(&self) -> usize {
        self.log_probs.len()
    }
}
