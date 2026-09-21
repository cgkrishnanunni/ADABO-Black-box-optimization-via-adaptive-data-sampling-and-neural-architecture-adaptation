# Black box optimization

Pip-installable surrogate-assisted black-box optimizer:

1. Fit a neural surrogate to labeled designs  
2. Optionally grow the network (topological layer insertion)  
3. Propose the next design by **Minimizing the surrogate** (exploitation), with
   **stagnation-triggered active learning** (exploration) when the best true loss
   stalls for `stagnation_m` successive minimization steps

You only supply a loss `f(x) -> float` in **physical** coordinates, box bounds,
and (optionally) an initial design of experiments.

Default demo: **4D Rosenbrock** on `[-2, 2]^4` (minimum at `(1,1,1,1)`).

---

## Requirements

- Python **3.9+**
- Dependencies (installed automatically): `numpy`, `torch`, `joblib`, `matplotlib`

---

## Setup (step by step)

```bash
cd /path/to/proposed_bbopt
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
python -c "from proposed_bbopt import ProposedOptimizer; print('OK')"
```

---

## Run the examples

```bash
jupyter notebook examples/optimize_rosenbrock.ipynb
```

---

## Quick start (4D Rosenbrock)

```python
import numpy as np
from proposed_bbopt import ProposedOptimizer

def rosenbrock(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )

bound = [(-2.0, 2.0)] * 4

result = ProposedOptimizer(
    fitness=rosenbrock,
    bound=bound,
    N_t_initial=5,
    N_t_total=100,
    stagnation_m=3,   # AL after 3 non-improving Cuckoo steps
    verbose=True,
    seed=0,
).execute()

print(result.best, result.best_fitness)
```

### Explicit `Hyperparameters`

```python
from proposed_bbopt import ProposedOptimizer, Hyperparameters

hyperp = Hyperparameters(
    vec_lower=np.array([-2.0, -2.0, -2.0, -2.0]),
    vec_upp=np.array([2.0, 2.0, 2.0, 2.0]),
    N_t_initial=5,
    N_t_total=100,
    stagnation_m=3,
    P=200,
    Tmax=500,
    iter=2000,
    Nh=20,
    sparsity=10,
    sparsity_input_Nh=np.array([0]),
    sparsity_input_I=np.array([1]),
    asymp_N=200,
    grid_opt=100,
    back_track_rate=0.1,
)

result = ProposedOptimizer(
    fitness=rosenbrock,
    hyperp=hyperp,
    verbose=True,
).execute()
```

---

## Acquisition policy

| Situation | Action |
|-----------|--------|
| Default | Cuckoo Search on the surrogate |
| Best true loss unchanged for `stagnation_m` Cuckoo steps | Next iteration uses low-rank active learning |
| After that AL step | Resume Cuckoo; stagnation counter resets |
| Best true loss improves | Stagnation counter resets |

---

## Important options

| Option | Default | Meaning |
|--------|---------|---------|
| `N_t_total` | 100 | Total labeled evaluations |
| `stagnation_m` | 3 | Non-improving Cuckoo steps before one AL kick |
| `Nh` | 20 | Hidden width |
| `sparsity` | 10 | Active neurons kept at init |
| `iter` | 2000 | Surrogate training epochs per outer iter |
| `grid_opt` | 100 | Grid searches inside AL |
| `back_track_rate` | 0.1 | Layer-insertion line-search rate |
| `P`, `Tmax` | 200, 500 | Cuckoo population / generations |

See `proposed_bbopt/hyperparameters.py` for the full list.

---

## Result fields

| Field | Meaning |
|-------|---------|
| `result.best` | Best physical design found |
| `result.best_fitness` | Corresponding loss |
| `result.history_fitness` | Best loss after each outer iteration |
| `result.history_solution` | Best physical design after each outer iteration |
| `result.X`, `result.y` | Full labeled dataset |

---
