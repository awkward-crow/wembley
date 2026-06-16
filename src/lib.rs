//! Histogram-based gradient boosting (LightGBM-style) in Rust.
//!
//! See the [repository](https://github.com/awkward-crow/wembley) for examples and benchmarks.

pub mod bin_mapper;
pub mod boosting;
pub mod config;
pub mod data_partition;
pub mod dataset;
pub mod histogram;
pub mod objective;
pub mod tree;
