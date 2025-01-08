# Experiments

To run Hyperparameter Tuning L0 experiments, follow the next steps:

1. Install requirements

```python
pip install -r requirements_versions_py_3_11.txt
```

2. Set up Wandb API KEY

```bash
export WANDB_API_KEY=<Your API KEY>
```

3. Run hyperparameter tuning script

```python
python3 experiments_l0_hyperparameter_tuning.py -p <dataset name>-ht -ht -ns <num of experiments> -d <dataset name> -k <num of k folds cross-validation> -e 400 -bs 128 --group_l0 --beta_ema 0.999
```

