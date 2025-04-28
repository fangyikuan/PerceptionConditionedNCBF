## Neural-CBF for 2D Driving model

### Quick-start guide & usage notes

---

### 1.  Environment & dependencies
```bash
# python ≥3.8  (3.8–3.11 tested)
conda create -n ncbf python=3.10
conda activate ncbf

# core libs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   # or cpu
pip install posggym==0.4.1 cvxpy tqdm matplotlib
```
> *CUDA* is optional – everything falls back to CPU.

---

### 2.  Repository layout
```
├── main.py                    # <─ the script shown below
├── unicycle.py                # Dubins-car dynamics
├── train_ncbf_new.py          # NCBFTrainer + CBFModel
├── trajectory_collection_parallel.py
├── dataset_collection.py      # TrajectoryDataset + normaliser helpers
└── Neural_CBF/
    └── cbf_model_epoch_10.pt  # sample trained weights
```

---

### 3.  Running the **demo / evaluation**
```bash
python test_driving_random.py
```
* `train = False` in **main.py** loads the pre-trained checkpoint
  `./Neural_CBF/cbf_model_epoch_10.pt` and shows one 1000-step rollout with the
  QP CBF-filter.  
* A window opens (`render_mode="human"`) – the car steers around random
  circular obstacles toward its destination.

---

### 4.  Training your own CBF network
1. **Collect 1 M transitions** (≈5 min @ 25 processes, 1000-step rollouts):
   ```python
   train = True                # flip the flag in main.py
   ```
   or run directly:
   ```bash
   python main.py --train          # see CLI below
   ```
   *Balanced sampling* is performed so “safe” and “unsafe” classes are equal.

2. **NCBF optimisation**  
   * hyper-parameters are passed to `NCBFTrainer` (see table below);
   * by default it trains for **10 epochs**, logging loss curves and saving
     `cbf_model_epoch_*.pt` after each epoch.

| Arg (in `NCBFTrainer`) | meaning | default |
|------------------------|---------|---------|
| `mu`           | regularisation weight (sigmoid) | `1` |
| `lambda_param` | weight of forward-invariance loss | `0.5` |
| `bound_eps`    | width of boundary band \|ϕ\|<ε for FI-loss | `0.2` |
| `total_epoch`  | epochs | `10` |
| `weight_decay` | AdamW/NaD​am L2 | `0` |

---

### 5.  Safe-action QP filter
`CBF_QP_Filter.safe_action`  
solves  

\[
\min_u \tfrac12\|u-u_\text{nom}\|^2 \quad
\text{s.t. } \dot h(x,u) + \alpha\,h(x)\ge 0,\;
u_\text{min}\le u\le u_\text{max}
\]

* **Nominal controller** – PIDGoalController (`unicycle.py`)
* **α** – set with `alpha` argument (default 0.1)
* **Box bounds** – `u_lower=(-0.25,-π/4)`, `u_upper=(0.25, π/4)`

If the QP becomes infeasible, the code returns the nominal action.

---

### 6.  Command-line flags (optional)
Modify **main.py** or call via environment variables:
```bash
python main.py \
  --seed 123 \
  --epochs 20 \
  --no-gpu
```
*(Wire-up left to you – easiest is `argparse` in the header.)*

---

### 7.  Tips & troubleshooting
* **BrokenPipeError** in multi-process collection → launch with  
  `python -X faulthandler main.py` and see `docs/fault-tolerant.md`
  (pattern: spawn + supervisor restart).
* To **record** videos, set  
  `render_mode="rgb_array"` in `env_kwargs` and dump frames in the roll-loop.
* **Hyper-parameters**: for tighter safety use larger `alpha` (QP) or `lambda_param`
  (trainer). For less conservative behaviour decrease them.

---

Enjoy experimenting with Neural Control Barrier Functions!