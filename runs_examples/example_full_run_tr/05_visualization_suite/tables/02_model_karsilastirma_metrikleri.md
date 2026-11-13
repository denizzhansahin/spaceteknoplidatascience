# Tablo 02: Gruplanmış Çapraz Doğrulama Model Kıyaslama Metrikleri

| Sıra | Model Adı | Model Ailesi | OOF RMSE (%) | OOF MAE (%) | Bias (%) | R² | RPIQ | LoA Alt / Üst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | kernel_ridge_rbf | kernel_method | 0.6083 | 0.4895 | +0.0283 | 0.7793 | 2.928 | [-1.170, +1.227] |
| 2 | plsr | linear_latent_chemometric | 0.6351 | 0.5080 | +0.0156 | 0.7595 | 2.804 | [-1.237, +1.268] |
| 3 | local_plsr_mbl | memory_based_learning | 0.6402 | 0.4883 | -0.0863 | 0.7556 | 2.782 | [-1.338, +1.165] |
| 4 | random_forest | tree_ensemble | 0.7069 | 0.5598 | -0.0019 | 0.7020 | 2.519 | [-1.396, +1.392] |
| 5 | st_soil_hybrid_net | neural_two_branch_physics_constrained | 0.7703 | 0.5710 | -0.0177 | 0.6461 | 2.312 | [-1.537, +1.501] |
