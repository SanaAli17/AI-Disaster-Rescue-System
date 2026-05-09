# AI-Disaster-Rescue-System

## Overview

The **AI Disaster Rescue System** is an intelligent hybrid disaster management and rescue simulation platform designed to support emergency response operations in dynamic and uncertain environments.

The system combines multiple Artificial Intelligence techniques including:

- Search Algorithms
- Constraint Satisfaction Problems (CSP)
- Machine Learning
- Fuzzy Logic
- Dynamic Replanning

The goal of the project is to simulate intelligent rescue operations where an autonomous rescue agent can:

- Select the most critical victim
- Find optimal rescue paths
- Avoid dangerous routes
- Handle uncertainty
- Reallocate limited resources dynamically
- Adapt to environmental changes in real-time

The project demonstrates how multiple AI techniques can work together to support smart emergency decision-making systems.

---

# Features

## Intelligent Victim Selection
- Selects the most suitable victim based on:
  - Severity
  - Distance
  - Risk level

## Pathfinding Algorithms
Implemented search algorithms include:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search
- Risk-Aware A* Search
- Greedy Best-First Search
- Hill Climbing

## Constraint Satisfaction Problem (CSP)
- Ambulance assignment using:
  - Backtracking
  - MRV-inspired heuristic
- Capacity constraints enforced
- Resource allocation optimization

## Machine Learning Integration
Three ML models are used for rescue priority prediction:

- k-Nearest Neighbors (kNN)
- Naive Bayes
- Decision Tree

## Fuzzy Logic for Uncertainty Handling
The system handles uncertainty using fuzzy inference based on:

- Distance
- Risk
- Severity
- Blockage probability

## Dynamic Replanning
The system adapts to real-time environmental changes such as:

- Road blockage
- Increased risk levels
- New victims
- Resource depletion

## Interactive Web Interface
- Environment visualization
- Rescue path visualization
- ML evaluation metrics
- Dynamic scenario simulation

---

# Functionalities

## Victim Prioritization
The rescue agent intelligently selects victims using:

```python
score = severity - (distance + risk)
```

## Risk-Aware Navigation
The agent can switch from shortest path planning to safer path planning when uncertainty increases.

## ML-Based Decision Support
Machine Learning models predict rescue urgency levels:

- HIGH
- MEDIUM
- LOW

using majority voting for final prediction.

## Fuzzy Decision-Making
Fuzzy logic allows flexible decision-making under uncertain conditions.

## Resource Allocation
Victims are assigned to ambulances using CSP-based resource allocation with constraints.

## Dynamic Environment Handling
The system supports environmental changes during runtime and recomputes decisions dynamically.

---

# Project Structure

```text
AI-Disaster-Rescue-System/
│
├── agent.py
├── app.py
├── csp.py
├── environment.py
├── fuzzy.py
├── logo.png
├── main.py
├── ml_model.py
├── requirements.txt
├── search.py
└── README.md
```

---

# File Descriptions

## `environment.py`
Creates the disaster environment including:
- Grid
- Victims
- Hospitals
- Risk zones
- Blocked roads

## `search.py`
Contains all pathfinding algorithms:
- BFS
- DFS
- A*
- Risk-Aware A*
- Greedy Best-First Search
- Hill Climbing

## `csp.py`
Implements:
- CSP resource allocation
- Backtracking
- MRV-inspired heuristic

## `ml_model.py`
Contains:
- ML dataset
- Model training
- Evaluation metrics
- Priority prediction

## `fuzzy.py`
Implements:
- Fuzzification
- Fuzzy inference rules
- Uncertainty handling

## `agent.py`
Handles:
- Victim selection
- Rescue decision-making
- Risk evaluation

## `app.py`
Web interface implementation for interactive visualization and simulation.

## `main.py`
Main integration file connecting all AI components together.

---

# Technologies Used

## Programming Language
- Python

## Libraries
- NumPy
- Scikit-learn
- Heapq
- Collections
- Streamlit / Flask (depending on deployment)

## AI Techniques
- Search Algorithms
- CSP
- Machine Learning
- Fuzzy Logic
- Dynamic Replanning

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-Disaster-Rescue-System.git
```

## Move Into Project Directory

```bash
cd AI-Disaster-Rescue-System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Main Simulation

```bash
python main.py
```

## Run Web Interface

```bash
python app.py
```

---

# Machine Learning Evaluation

The system evaluates ML models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Dynamic Scenarios

The project supports multiple real-time dynamic scenarios:

## Scenario 1: Road Blockage
- Recomputes path using A*

## Scenario 2: Risk Increase
- Switches to Risk-Aware A*

## Scenario 3: New Victim
- Re-evaluates victim priorities

## Scenario 4: Resource Depletion
- Reassigns victims using CSP

---

# Future Improvements

Possible future extensions include:

- Real-time map integration
- Multi-agent coordination
- Reinforcement Learning
- Deep Learning-based prediction
- Live disaster sensor integration
- GPS integration
- Real-time traffic data

---

# Authors

## Team Members

- Sana Ali
- Syeda Kashaf Zahra
---


# Conclusion

This project demonstrates how multiple Artificial Intelligence techniques can be integrated together to create an intelligent disaster rescue and emergency response system capable of dynamic decision-making under uncertain conditions.
