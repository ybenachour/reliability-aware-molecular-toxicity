# Reliability-Aware Multi-Endpoint Molecular Toxicity Prediction under Scaffold Shift

This repository contains the scientific source code and curated reproducibility artifacts supporting the study **Reliability-Aware Multi-Endpoint Molecular Toxicity Prediction under Scaffold Shift**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21921230.svg)](https://doi.org/10.5281/zenodo.21921230)

The study evaluates six binary molecular assay endpoints: hERG blockade, Ames mutagenicity, and the Tox21 SR-p53, SR-ATAD5, SR-ARE, and SR-MMP endpoints. The workflow compares classical QSAR, neural-network, graph, and multitask models and evaluates scaffold-aware generalization, probability calibration, applicability-domain behavior, uncertainty, selective prediction, activity cliffs, and independent external validation.

## Release scope

This directory is the **v1.0.0 scientific and reproducibility release** intended for GitHub and Zenodo archival. The commercial **MolVerity/ToxVerity** interface is maintained separately and is **not included** in this repository.

## Repository contents

- `src/` — reusable scientific Python package.
- `configs/` — workflow configuration files.
- `scripts/` — workflow utilities.
- `tests/` — software tests.
- `notebooks/` — numbered scientific analysis workflow from `00` through `26`.
- `figures/jcim_manuscript_pdf/` — final vector manuscript figures.
- `reproducibility/` — curated provenance records, split assignments, aggregate metrics, external-validation outputs, manuscript tables, and reproducibility audit artifacts.

## Data provenance

Raw source datasets are not redistributed in this repository. Acquisition information, checksums, source URLs, endpoint definitions, and recorded licensing notes are provided in:

`reproducibility/metadata/dataset_registry.csv`

Users should consult the original data providers and their current terms before obtaining or redistributing source data.

## Trained models

Large trained model binaries are not included in the Git repository. The training, calibration, evaluation, and reproducibility code is provided so the documented analyses can be regenerated from the referenced source data and configuration.

## Reproducibility anchors

Primary split assignments:

`reproducibility/splits/split_assignments.csv`

Reproducibility audit:

`reproducibility/reports/reproducibility_audit.json`

Additional locked analysis outputs are retained in the corresponding `reproducibility/` subdirectories.

## Environment

The project targets Python 3.11. Reproducible environment files are provided in `environment.yml`, `requirements.txt`, and `requirements-windows.txt`.

Typical setup:

```bash
conda env create -f environment.yml
conda activate toxicity-screening
python -m pip install -e ".[ml,deep,data,dev]"
python -m pytest -q
```

The Makefile also provides convenience targets:

```bash
make install
make test
make lint
```

## Workflow

The analysis notebooks are numbered from `00` through `26`. For a complete regeneration from source data, run them in numerical order. See `notebooks/README.md` for the workflow description.

The complete workflow may generate large local datasets, feature files, model binaries, and intermediate outputs that are intentionally excluded from Git. The curated `reproducibility/` snapshot documents the locked analysis artifacts supporting the manuscript.

## Citation

Software archive:

**Benachour, Y. (2026). Reliability-Aware Molecular Toxicity Prediction (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21921230**

## License

The original software in this repository is released under the **BSD 3-Clause License**; see `LICENSE`. Third-party datasets, dependencies, and publications remain subject to their own terms.
