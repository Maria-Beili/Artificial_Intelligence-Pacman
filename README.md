## Artificial Intelligence Projects (UPF) - Pacman AI Agents

### Project Overview

This repository includes four practical AI projects developed for the Artificial Intelligence course at Universitat Pompeu Fabra. All tasks are based on the [Berkeley Pacman AI framework](https://inst.eecs.berkeley.edu/~cs188/sp22/projects/) and span a wide range of foundational AI topics, including search algorithms, adversarial reasoning, reinforcement learning, and multi-agent coordination. Each project is implemented in Python 3.8 and developed using the infrastructure provided by Berkeley's CS188.

### Contents

#### 1. **Search Agents** (`01_SingleAgent_Search`)

* **Objective**: Implement uninformed and informed single-agent search algorithms (DFS, BFS, UCS, A\*).
* **Main File**: `search.py`, `searchAgents.py`
* **Concepts**: State space representation, heuristics, optimality.

#### 2. **Adversarial Agents** (`02_Adversarial_MultiAgent`)

* **Objective**: Implement multi-agent search strategies with adversarial logic (Minimax, Alpha-Beta pruning).
* **Main File**: `multiAgents.py`
* **Concepts**: Game trees, evaluation functions, opponent modeling.

#### 3. **Reinforcement Learning** (`03_Reinforcement_Learning`)

* **Objective**: Develop agents that learn through trial-and-error using Q-learning and value iteration.
* **Main Files**: `valueIterationAgents.py`, `qlearningAgents.py`, `analysis.py`
* **Concepts**: MDPs, Bellman equation, exploration vs exploitation.

#### 4. **Pacman Capture the Flag Contest** (`04_CaptureTheFlag_TeamAgents`)

* **Objective**: Design intelligent agents for a team-based Pacman capture-the-flag game.
* **Main File**: `my_team.py`
* **Concepts**: Multi-agent coordination, real-time planning, agent design, team strategies.

### How to Run

For each project:

```bash
cd project_directory
python3.8 pacman.py -p <AgentClass> -l <layout> -z 0.5
```

For the final contest:

```bash
cd src/contest
python capture.py -r agents/my_team/my_team.py -b baselineTeam --record --record-log
```

### Dependencies

* Python 3.8
* Berkeley Pacman codebase
* No external libraries unless explicitly stated in the project instructions.

---

## Proyectos de Inteligencia Artificial (UPF) - Agentes Pacman

### Descripción General

Este repositorio contiene cuatro proyectos prácticos desarrollados en el curso de Inteligencia Artificial de la Universitat Pompeu Fabra. Los ejercicios se basan en el entorno de Pacman de la Universidad de Berkeley e integran conceptos clave como búsqueda, juegos adversariales, aprendizaje por refuerzo y sistemas multi-agente.

### Contenido

#### 1. **Agentes de Búsqueda** (`01_SingleAgent_Search`)

* **Objetivo**: Implementar algoritmos de búsqueda como DFS, BFS, UCS y A\*.
* **Archivos**: `search.py`, `searchAgents.py`
* **Conceptos**: Representación de estados, heurísticas, optimalidad.

#### 2. **Agentes Adversarios** (`02_Adversarial_MultiAgent`)

* **Objetivo**: Implementar estrategias de búsqueda multi-agente con lógica adversarial (Minimax, poda Alfa-Beta).
* **Archivo principal**: `multiAgents.py`
* **Conceptos**: Árboles de juego, funciones de evaluación, modelado de oponentes.

#### 3. **Aprendizaje por Refuerzo** (`03_Reinforcement_Learning`)

* **Objetivo**: Crear agentes que aprendan mediante interacción con el entorno (Q-learning y value iteration).
* **Archivos**: `valueIterationAgents.py`, `qlearningAgents.py`, `analysis.py`
* **Conceptos**: Procesos de decisión de Markov, ecuación de Bellman.

#### 4. **Concurso Pacman Capture the Flag** (`04_CaptureTheFlag_TeamAgents`)

* **Objetivo**: Diseñar agentes inteligentes para competir en modo "capture the flag" en equipos.
* **Archivo principal**: `my_team.py`
* **Conceptos**: Coordinación multi-agente, planificación en tiempo real.

### Ejecución

Para cada proyecto:

```bash
cd directorio_del_proyecto
python3.8 pacman.py -p <NombreAgente> -l <mapa> -z 0.5
```

Para el concurso:

```bash
cd src/contest
python capture.py -r agents/my_team/my_team.py -b baselineTeam --record --record-log
```

### Dependencias

* Python 3.8
* Infraestructura Pacman CS188
* No se requieren librerías externas salvo indicación explícita

---

### Autores

Trabajo en grupo para la asignatura de Inteligencia Artificial (Universitat Pompeu Fabra).
Responsables: \[Nombres de los autores aquí]

Este repositorio sirve como demostración técnica del conocimiento aplicado en el diseño de agentes inteligentes.
