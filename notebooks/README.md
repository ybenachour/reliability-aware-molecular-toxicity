# Toxicity Screening Notebooks

This folder contains the complete numbered scientific workflow for the reliability-aware molecular toxicity study.

1. **`00_project_overview.ipynb`** — Defines the project objectives, toxicity endpoints, intended use, expected inputs and outputs, limitations, and complete research workflow.
2. **`01_environment_and_reproducibility.ipynb`** — Verifies the Python environment, package versions, hardware availability, random seeds, directory structure, and reproducibility settings.
3. **`02_dataset_acquisition.ipynb`** — Downloads the hERG, Ames, and Tox21 datasets from configured sources while preserving the unmodified raw files and checksums locally.
4. **`03_data_provenance_and_endpoint_definitions.ipynb`** — Documents dataset source, licensing/provenance information, version, assay context, label definition, missingness, and endpoint-specific dataset cards.
5. **`04_molecular_standardization.ipynb`** — Standardizes raw molecular structures into consistent canonical representations while recording transformations and failures.
6. **`05_data_quality_and_duplicate_audit.ipynb`** — Audits invalid molecules, duplicate structures, conflicting labels, class distributions, missing labels, and duplicate-resolution policies.
7. **`06_exploratory_data_analysis.ipynb`** — Summarizes endpoint prevalence, molecular descriptors, scaffold diversity, label missingness, and molecular-property distributions.
8. **`07_chemical_space_and_scaffold_analysis.ipynb`** — Examines descriptor-space structure, fingerprint similarity, scaffold frequencies, endpoint overlap, analogue series, and chemical-space diversity.
9. **`08_split_generation.ipynb`** — Creates persistent random, global scaffold-aware, and repeated scaffold splits while preventing molecule and scaffold leakage.
10. **`09_feature_generation.ipynb`** — Generates molecular descriptors, Morgan fingerprints, molecular graph features, and stable molecule-to-feature indices.
11. **`10_qsar_baselines.ipynb`** — Trains and evaluates endpoint-specific classical QSAR models under the primary scaffold split.
12. **`11_single_task_neural_models.ipynb`** — Trains separate fingerprint-based neural networks for each toxicity endpoint.
13. **`12_single_task_gnn_models.ipynb`** — Trains separate graph neural networks for each endpoint using molecular graph representations.
14. **`13_multitask_fingerprint_model.ipynb`** — Trains a shared fingerprint-based neural network with endpoint-specific heads and masked losses for missing labels.
15. **`14_multitask_gnn_model.ipynb`** — Trains a shared molecular-graph encoder with endpoint-specific prediction heads using masked multitask learning.
16. **`15_hyperparameter_optimization.ipynb`** — Optimizes selected model hyperparameters using a fixed validation protocol and controlled Optuna search budget.
17. **`16_calibration_and_uncertainty.ipynb`** — Compares calibration and uncertainty procedures and selects endpoint-specific probability-calibration strategies using validation data.
18. **`17_applicability_domain_and_ood.ipynb`** — Quantifies applicability-domain membership, scaffold novelty, chemical distance, out-of-distribution status, and abstention criteria.
19. **`18_activity_cliff_analysis.ipynb`** — Identifies structurally similar molecules with discordant toxicity labels and quantifies activity-cliff burden.
20. **`19_interpretability.ipynb`** — Produces model explanations and evaluates their stability and limitations.
21. **`20_external_validation.ipynb`** — Performs locked external evaluation for the scientifically compatible Tox21 endpoints after exact molecule-overlap auditing and exclusion, including bootstrap uncertainty and selective-prediction analyses.
22. **`21_ablation_and_negative_transfer.ipynb`** — Measures feature/model ablations and positive or negative multitask transfer.
23. **`22_final_model_comparison.ipynb`** — Compares eligible models using predictive discrimination, calibration, reliability, and robustness criteria.
24. **`23_candidate_screening_demo.ipynb`** — Demonstrates research-grade endpoint-specific screening for individual SMILES strings or CSV inputs using selected calibrated models when the required model bundles are available locally.
25. **`24_reproducibility_audit.ipynb`** — Verifies datasets, splits, results, metadata, tests, software-version records, and deterministic inference requirements used by the study.
26. **`25_manuscript_tables_and_figures.ipynb`** — Generates manuscript-oriented tables, summaries, and publication outputs from the locked analyses.
27. **`26_manuscript_figures_v2.ipynb`** — Regenerates the final JCIM main-text and supporting manuscript figures from locked analysis outputs, including the refreshed external-validation figure.

## Recommended execution order

For a complete regeneration from source data, run the notebooks sequentially from **`00` through `26`** because later stages depend on artifacts produced by earlier notebooks. Some notebooks download public source datasets or generate large local artifacts that are intentionally not redistributed in this repository; consult the root `README.md` and `reproducibility/metadata/dataset_registry.csv` before execution.
