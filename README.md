# SAREnv – Radiation-Aware UAV Search and Rescue Extension

This repository contains my MSc dissertation development based on the open-source [SAREnv](https://github.com/namurproject/SAREnv) framework.

The original SAREnv project provides a dataset and evaluation framework for UAV-based wilderness Search and Rescue (SAR).

For my **MSc Aerial Robotics dissertation at the University of Bristol**, I extended SAREnv with a radiological-hazard layer and developed a **radiation-aware search strategy** to investigate the trade-off between search performance and radiation exposure in hazardous environments.

> **My work focuses on UAV mission simulation, algorithm development, environmental modelling, large-scale validation and performance optimisation.**

---

## My MSc Dissertation Contribution

The original SAREnv framework does not model radiological hazards.

My dissertation extends the framework with a complete radiation-aware simulation and evaluation pipeline, including:

- Radiation-field modelling for multiple hazard geometries
- Online radiation estimation from simulated measurements
- Radiation-aware search prioritisation
- Trigger and hysteresis logic for switching search behaviour
- UAV and survivor radiation-exposure evaluation
- Multi-scenario and multi-seed experiment infrastructure
- Radiation-intensity parameter studies
- Automated statistical analysis and result generation
- Performance optimisation of high-resolution trajectory evaluation
- Regression tests and validation tools

The objective of the project is **not radiation-source hunting**.

Instead, the work investigates how environmental radiation information can be integrated into an existing UAV search-and-rescue mission so that high-risk areas can be prioritised while preserving overall search effectiveness.

---

## System Overview

The extended mission pipeline can be summarised as:

```text
SAR Environment
      ↓
Lost-Person Probability Model
      ↓
Radiation Environment
      ↓
Simulated UAV Measurements
      ↓
Online Radiation Estimation
      ↓
Radiation-Aware Search Priority
      ↓
UAV Search Route
      ↓
Mission Evaluation
      ↓
Search Performance + Radiation Exposure
```

The project therefore connects:

**environment modelling → sensing / estimation → mission decision logic → trajectory execution → quantitative evaluation**

---

## Radiation Environment Modelling

Three radiation-field structures were implemented and evaluated.

### Single Point Source

A localised radiation source represented using a softened inverse-square radiation field.

### Multiple Point Sources

Multiple separated sources create several competing high-risk regions and a more complex spatial radiation distribution.

### Surface Source

A distributed radiation source produces a broader and smoother radiation field.

These scenarios were used to investigate how different spatial hazard structures affect search behaviour.

---

## Online Radiation Estimation

The planner does not simply receive the complete radiation ground truth.

Instead, simulated UAV measurements are used to update an online radiation estimate during the mission.

The estimator allows the search strategy to react to newly observed radiation information while maintaining the original SAR probability model.

This produces a simplified perception–decision loop:

```text
UAV Position
     ↓
Radiation Measurement
     ↓
Local Estimate Update
     ↓
Search-Priority Adjustment
     ↓
Next Search Decision
```

---

## Radiation-Aware Search Strategy

The baseline SAREnv **Greedy Search** prioritises search regions according to the lost-person probability distribution.

My radiation-aware extension preserves the existing SAR logic while adding radiation information as an additional prioritisation signal.

The planner therefore balances two competing objectives:

- Search areas where survivors are likely to be located
- Prioritise survivors who may be experiencing higher radiation exposure

This provides a controlled way to study the trade-off between:

**SAR efficiency ↔ radiation-risk reduction**

without replacing the original mission architecture.

---

## Trigger and Hysteresis Logic

A trigger mechanism determines when radiation information should influence search priority.

Hysteresis is used to reduce rapid switching around the activation threshold.

This avoids unstable planner behaviour caused by small changes in estimated radiation levels and provides more consistent mission decisions.

---

## Mission and Exposure Evaluation

The evaluation system was extended to quantify both conventional SAR performance and radiological risk.

### Search Performance

- Survivor detection
- Search likelihood
- Time-discounted search performance
- Mission progress
- Route behaviour

### Radiation Exposure

- Survivor cumulative radiation exposure
- UAV radiation exposure
- Total exposure
- High-exposure population metrics
- P95 exposure
- Maximum individual exposure

The combined metrics allow direct comparison between the baseline and radiation-aware search strategies.

---

## Experimental Design

The final validation study used:

- **20 random seeds**
- **5 radiation intensities**
- **3 radiation-field structures**
  - Single
  - Multi
  - Surface

This produced **300 paired scenario evaluations** comparing:

```text
Original Greedy
       vs
Radiation-Aware Greedy
```

The paired design ensures that both planners are evaluated under the same underlying scenario conditions.

---

## Representative Findings

The experiments showed that radiation-aware planning can reduce high-risk survivor exposure, although the effect depends strongly on the spatial structure of the radiation field.

### Single Source

Radiation-aware search generally reduced survivor radiation exposure while introducing a search-performance trade-off.

### Surface Source

The distributed hazard structure produced the most consistent reduction in exposure across the tested scenarios.

### Multiple Sources

The multi-source environment showed substantially larger variability.

Several separated radiation hotspots can compete for search priority, causing larger changes in search order and occasionally producing worse individual worst-case outcomes.

This provided an important failure case for analysing planner behaviour and the limitations of the current priority-based strategy.

---

## Large-Scale Validation and Failure Analysis

A major part of the dissertation was not only implementing the algorithm, but also building the infrastructure required to test it systematically.

The experiment pipeline supports:

- Repeated seeded trials
- Paired planner comparisons
- Radiation-intensity sweeps
- Multiple spatial hazard models
- Automatic CSV result generation
- Statistical summaries
- Plot and table generation
- Cross-seed variability analysis

Unexpected results were investigated rather than discarded.

In particular, the Multi-source scenario was analysed to understand why competing radiation hotspots could cause unstable search ordering and large variation in maximum survivor exposure.

---

## Performance Optimisation

High-resolution radiation evaluation originally became a major computational bottleneck.

Radiation exposure was integrated along UAV trajectories using approximately **1 m spatial sampling**.

The original implementation repeatedly generated trajectory samples using scalar operations, resulting in very high execution time.

I refactored the integration pipeline using NumPy-based vectorised trajectory generation and reusable evaluator caches.

For a representative trajectory containing **87,211 identical sampling points**:

| Evaluation Stage | Original | Optimised |
|---|---:|---:|
| 1 m coordinate generation | 376.867 s | 0.0119 s |
| Full radiation integration | 379.375 s | 1.096 s |
| Overall speed-up | — | **~346×** |

Regression checks confirmed that the optimisation preserved:

- Sample positions
- Route geometry
- Evaluation spacing
- Radiation integration behaviour

This optimisation made repeated large-scale validation practical.

---

## Engineering and Research Skills Demonstrated

This project involved substantially more than running an existing simulation.

### Algorithm Development

- Radiation-aware search prioritisation
- Risk-aware decision logic
- Trigger / hysteresis design
- Planner comparison

### Simulation and Modelling

- UAV search-and-rescue simulation
- Radiation-field modelling
- Online environmental estimation
- Mission trajectory evaluation

### Software Engineering

- Python
- NumPy
- Existing-codebase extension
- Modular integration
- Performance profiling and optimisation
- Regression testing
- Git-based development

### Experimentation

- Controlled paired experiments
- Parameter sweeps
- Random-seed validation
- Automated experiment runners
- Statistical analysis
- Failure-case investigation

### UAV / Robotics Context

- Mission planning
- Search strategy
- Environment-aware decision making
- Trajectory evaluation
- Simulated sensor information
- Algorithm validation

---

## Repository Structure

The repository is based on the upstream SAREnv architecture.

The dissertation extensions are primarily contained in the radiation-related modules, experiment scripts and associated tests.

```text
SAREnv/
├── sarenv/                 # Core SAREnv package and extensions
├── examples/
│   └── radiation/          # Radiation-aware experiments and evaluation scripts
├── tests/                  # Original and dissertation-related tests
├── results/                # Selected experiment outputs
├── data/                   # Supporting data
├── sarenv_dataset/         # SAR environment datasets
└── README.md
```

For the exact implementation history, please refer to the Git commit history.

---

## Original SAREnv Project

This work is built on the open-source **SAREnv** framework.

Original repository:

https://github.com/namurproject/SAREnv

The original framework provides:

- UAV wilderness-search datasets
- Lost-person probability modelling
- Baseline search planners
- Search-path evaluation
- Comparative algorithm benchmarking

My dissertation work extends this framework rather than replacing the original SAREnv architecture.

---

## About This Fork

This fork serves two purposes:

1. Preserve the original SAREnv framework and development history
2. Document the implementation developed for my MSc dissertation

The radiation-aware extensions described above represent my dissertation contribution.

The original SAREnv code and intellectual contributions remain attributed to the upstream authors.

---

## Author

**Kangping Wu**

MSc Aerial Robotics  
University of Bristol

Research interests:

- UAV systems
- Robotics
- Mission planning
- Simulation and validation
- Search and path-planning algorithms
- Environment-aware autonomous systems

GitHub: [1685104738-creator](https://github.com/1685104738-creator)

---

## License

This repository retains the licence of the upstream SAREnv project.

See [`LICENSE`](LICENSE) for details.
