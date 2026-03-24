use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, ToPyArray};
use ndarray::Array1;
use rayon::prelude::*;

#[pyfunction]
pub fn compute_gae_batch<'py>(
    py: Python<'py>,
    rewards_list: &PyList,
    values_list: &PyList,
    dones_list: &PyList,
    gamma: f32,
    gae_lambda: f32,
) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
    let all_rewards: Vec<Vec<f32>> = rewards_list.extract().unwrap();
    let all_values: Vec<Vec<f32>> = values_list.extract().unwrap();
    let all_dones: Vec<Vec<f32>> = dones_list.extract().unwrap();

    let mut all_advs = Vec::new();
    let mut all_rets = Vec::new();

    let num_trajectories = all_rewards.len();
    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..num_trajectories)
        .into_par_iter()
        .map(|i| {
            compute_single_gae(
                &all_rewards[i],
                &all_values[i],
                &all_dones[i],
                gamma,
                gae_lambda,
            )
        })
        .collect();

    for (advs, rets) in results {
        all_advs.extend(advs);
        all_rets.extend(rets);
    }

    let adv_array = Array1::from_vec(all_advs);
    let ret_array = Array1::from_vec(all_rets);

    (adv_array.to_pyarray(py), ret_array.to_pyarray(py))
}

fn compute_single_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut returns = vec![0.0; n];
    let mut gae = 0.0;

    for i in (0..n).rev() {
        let next_value = if i == n - 1 {
            if dones[i] >= 1.0 { 0.0 } else { values[i] }
        } else {
            if dones[i] >= 1.0 { 0.0 } else { values[i + 1] }
        };

        let delta = rewards[i] + gamma * (1.0 - dones[i]) * next_value - values[i];
        gae = delta + gamma * gae_lambda * (1.0 - dones[i]) * gae;

        advantages[i] = gae;
        returns[i] = gae + values[i];
    }

    (advantages, returns)
}
