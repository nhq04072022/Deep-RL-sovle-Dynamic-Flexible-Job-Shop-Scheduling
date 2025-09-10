# Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling

## Overview

This repository contains the source code for a shop-floor simulation environment, deep multi-agent reinforcement learning (MARL) algorithms for job sequencing and machine routing, and experiment setups. The goal is to provide a benchmark framework for comparing scheduling methods in dynamic environments (e.g., job arrivals, machine breakdowns).

---

## Repository Structure

* `agent/` — defines shop-floor assets such as machines, work centers, and equipment states.
* `brain/` — learning modules (routing agent, sequencing agent) and state/action functions.
* `creation/` — generates dynamic events, including job arrivals, machine breakdowns, and repairs.
* `main/` — main processes for running simulations, training, or validation.
* `routing/`, `sequencing/` — classical priority rules for baseline comparison.
* `validation/` — modules for importing trained models and running evaluation experiments.

Result and model directories:

* `experiment_result/` — experiment outputs (logs, metrics, reports).
* `routing_models/` — trained routing agent checkpoints.
* `sequencing_models/` — trained sequencing agent checkpoints.

---

## Usage

* **Sequencing Agent (SA):**

  * Network builder class: `network_validated` in `brain_machine_S.py`.

* **Routing Agent (RA):**

  * Network builder classes: `build_network_small`, `build_network_medium`, `build_network_large` in `brain_workcenter_R.py`.

* State and action functions are defined in both `brain_machine_S.py` and `brain_workcenter_R.py`.

---

## Training — Train Agents

Two main scripts are provided for training:

* `main_training_R.py` — trains the Routing Agent (RA).
* `main_training_S.py` — trains the Sequencing Agent (SA).

---

## Validation — Run Experiments

Scripts for running validation experiments with trained models:

* `main_experiment_R.py` — validation/experiments for routing.
* `main_experiment_S.py` — validation/experiments for sequencing.
* `main_experiment_integrated.py` — experiments with integrated routing and sequencing.

When running validation, specify the path to the trained checkpoint/model as an argument (e.g., `--model_path` or equivalent supported by the script).

---

## Test Scenarios

The following benchmark scenarios are included for evaluation:

1. **HH – High Heterogeneity, Tight Deadlines:** job size range \[5, 25], due date factor \[1, 2].
2. **HL – High Heterogeneity, Loose Deadlines:** job size range \[5, 25], due date factor \[1, 3].
3. **LH – Low Heterogeneity, Tight Deadlines:** job size range \[10, 20], due date factor \[1, 2].
4. **LL – Low Heterogeneity, Loose Deadlines:** job size range \[10, 20], due date factor \[1, 3].

---
