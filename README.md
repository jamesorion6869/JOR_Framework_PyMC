# JOR-Framework-PyMC

PyMC implementation of the **James Orion Report (JOR) Framework** for UAP case analysis.  
This repository contains scripts to calculate **SOP**, **NHP**, and posterior probabilities for JOR Framework v3 cases using Bayesian methods.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.2-orange.svg)](https://www.pymc.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Workflow Overview

1. Run `jor_fusion.py` → generates `jor_scores.csv` with SOP/NHP for each case  
2. Run `jor_pymc_runner.py` → reads `jor_scores.csv`, performs Bayesian analysis, updates it with posterior means and distributions

```
jor_fusion.py
      ↓
   jor_scores.csv (initial SOP/NHP)
      ↓
jor_pymc_runner.py
      ↓
   jor_scores.csv (updated with posterior means and distributions)
```

---

## Repository Contents

- `jor_fusion.py` – Generates `jor_scores.csv` with SOP/NHP scores  
- `jor_pymc.py` – Main PyMC Bayesian model for JOR cases  
- `jor_pymc_runner.py` – Reads `jor_scores.csv` and updates it with posterior means and distributions

---

## Requirements

- Python 3.10+  
- [PyMC](https://www.pymc.io/)  
- NumPy  
- Pandas  

Install dependencies with pip:

```
pip install pymc numpy pandas
```

---

## How to Run

1. Generate initial case scores:

```
python jor_fusion.py
```

2. Run Bayesian analysis:

```
python jor_pymc_runner.py
```

3. Outputs:

- `jor_scores.csv` updated with:
  - Original SOP and NHP values  
  - Posterior means  
  - Posterior distributions

4. Optional: Call `jor_pymc.py` directly for integration or custom analysis.

---

## Notes

- NHP is conditionally dependent on SOP according to JOR Framework logic  
- This repo is a self-contained PyMC workflow for JOR case analysis  
- Future updates may include additional automation for case imports and helper scripts

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)
