# Projected-Target Coresets

This repository contains the implementation of the paper entitled "Efficient Dataset Condensation for Time-Series Forecasting via Target Projection," currently under review at PVLDB. It includes PTC dataset construction, the Random baseline, and downstream evaluation with DLinear, iTransformer, and PatchTST.

## Methods

PTC selects historical input windows using regularized leverage-score sampling and assigns targets obtained from a shared linear projection fitted on the complete training split. Random samples input windows uniformly without replacement and retains their observed targets.


## Requirements

The paper experiments used Python 3.12.3, PyTorch 2.7.0, CUDA 12.9, and cuDNN 9.9.0. Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Download the public long-term forecasting datasets and arrange them below one directory as follows:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── exchange_rate/exchange_rate.csv
├── weather/weather.csv
├── electricity/electricity.csv
└── traffic/traffic.csv
```

The code uses the standard chronological long-term forecasting protocol. ETT datasets use fixed 12/4/4-month train/validation/test splits. Exchange, Weather, Electricity, and Traffic use 70%/10%/20% chronological splits. Standardization statistics are fitted on the training split only. The input length is 96 and the prediction horizons are 96, 192, 336, and 720.

## Quick reproduction run

The quickest end-to-end check reproduces the ETTm2 rows at `K=10` with DLinear students. It evaluates both Random and PTC over four prediction horizons and five student seeds:

```bash
python run_experiment.py --scope check --data-root /path/to/dataset --output-root results_check --device cuda
```

The command writes the individual runs and their aggregate statistics to `results_check/`. The primary output is `results_check/four_horizon_summary.csv`, which contains MSE and MAE averaged over the four horizons and five student seeds.

## Full main experiment

To run the complete main grid on one GPU:

```bash
python run_experiment.py --scope full --data-root /path/to/dataset --output-root results_full --device cuda
```

The full grid contains eight datasets, four horizons, two compression budgets (`K=10,25`), two methods, three student architectures, and five seeds. It creates 32 construction configurations and 1,920 downstream training runs. Runs are executed sequentially so the command works on a single GPU. A completed student run is skipped automatically when the command is restarted.

The main outputs are:

```text
results_full/
├── artifacts/                  # constructed compressed datasets
├── runs/                       # individual student results
├── all_runs.csv                # every completed run
├── per_horizon_summary.csv     # seed-averaged results
└── four_horizon_summary.csv    # results averaged over all horizons
```

## Running individual stages

Construct Random and PTC datasets for one configuration:

```bash
python construct.py --config ETTm2_H96 --data-root /path/to/dataset --output-root results --budgets 10 25 --device cuda
```

Train one student on one constructed artifact:

```bash
python train.py --config ETTm2_H96 --artifact results/artifacts/ETTm2_H96/K10/ptc.pt --student DLinear --seed 0 --data-root /path/to/dataset --output results/runs/ETTm2_H96/K10/ptc/DLinear/seed0 --device cuda
```

Aggregate completed student runs:

```bash
python summarize.py --result-root results
```
