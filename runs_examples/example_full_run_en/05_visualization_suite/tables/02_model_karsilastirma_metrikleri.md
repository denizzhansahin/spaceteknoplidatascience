# Table 02: Grouped Cross-Validation Model Benchmark Metrics

| Rank | Model Name | Family | OOF RMSE (%) | OOF MAE (%) | Bias (%) | R² | RPIQ | LoA Low / High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | kernel_ridge_rbf | kernel_method | 0.2577 | 0.1893 | -0.0117 | 0.9625 | 7.881 | [-0.520, +0.496] |
| 2 | plsr | linear_latent_chemometric | 0.3400 | 0.2575 | +0.0039 | 0.9347 | 5.974 | [-0.667, +0.674] |
| 3 | local_plsr_mbl | memory_based_learning | 0.4016 | 0.3014 | +0.0607 | 0.9089 | 5.058 | [-0.722, +0.844] |
| 4 | st_soil_hybrid_net | neural_two_branch_physics_constrained | 0.6578 | 0.5018 | -0.0580 | 0.7556 | 3.088 | [-1.350, +1.234] |
| 5 | random_forest | tree_ensemble | 0.6678 | 0.5347 | +0.0133 | 0.7481 | 3.042 | [-1.304, +1.330] |
