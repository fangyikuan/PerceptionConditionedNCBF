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
cd .. # make sure you're returning to the PerceptionConditionedNCBF directory
pip install -e .
pip install cvxpy tqdm matplotlib
```
> *CUDA* is optional – everything falls back to CPU.

---

### 2.  Repository layout
```
Neural_CBF
├── driving_random.py          # <─ the main and the only script to run
├── unicycle.py                # Dubins-car dynamics
├── train_ncbf_new.py          # NCBFTrainer + CBFModel
├── trajectory_collection_parallel.py
├── dataset_collection.py      # TrajectoryDataset + normaliser helpers
```

---

### 3.  Running the **training script**
```bash
python driving_random.py --train --num_traj 10000 --dump
```
* `--train` in **driving_random.py** automatically initialize environment, perform traj sampling, training and evaluation model.
* After Training, the checkpoints for Neural CBFs will be shown as `cbf_model_epoch_{xx}.pt`
* A window opens (`render_mode="human"`) – the car steers around random
  circular obstacles toward its destination.

---


### 4. **NCBF optimisation**  
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
Modify **driving_random.py** or call via environment variables:
```bash
python driving_random.py \
  --seed 123 \
  --epochs 20 \
  --no-gpu
```
*(Wire-up left to you – easiest is `argparse` in the header.)*

---

Enjoy experimenting with Neural Control Barrier Functions!