use pyo3::prelude::*;

mod buffer;
mod gae;

use buffer::RustReplayBuffer;
use gae::compute_gae_batch;

#[pymodule]
fn rust_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustReplayBuffer>()?;
    m.add_function(wrap_pyfunction!(compute_gae_batch, m)?)?;
    Ok(())
}
