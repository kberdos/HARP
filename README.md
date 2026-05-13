# HARP: Transferable Neural WAN TE for Changing Topologies
[HARP](https://dl.acm.org/doi/10.1145/3651890.3672237) is a transferable neural network for WAN Traffic Engineering that is designed to handle changing topologies. It was published at ACM SIGCOMM 2024.

If you use this code, please cite:
```
@inproceedings{HARP,
author = {AlQiam, Abd AlRhman and Yao, Yuanjun and Wang, Zhaodong and Ahuja, Satyajeet Singh and Zhang, Ying and Rao, Sanjay G. and Ribeiro, Bruno and Tawarmalani, Mohit},
title = {Transferable Neural WAN TE for Changing Topologies},
year = {2024},
isbn = {9798400706141},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3651890.3672237},
doi = {10.1145/3651890.3672237},
booktitle = {Proceedings of the ACM SIGCOMM 2024 Conference},
pages = {86–102},
numpages = {17},
keywords = {traffic engineering, wide-area networks, network optimization, machine learning},
location = {Sydney, NSW, Australia},
series = {ACM SIGCOMM '24}
}
```
Please contact `aalqiam@purdue.edu` for any questions.
### Environment Used
HARP was tested using the following setup:
- Ubuntu 22.04 machine
- Python 3.10.6
- `torch==2.1.0+cu121`
- `torch-scatter==2.1.2`
- Check the rest in requirements.txt
### Required Libraries
1. Install the required Python packages as listed in the requirements.txt. Use:
   `pip3 install -r requirements.txt`
2. Please follow this [link](https://pytorch.org/get-started/locally/) to install a version of PyTorch that fits your environment (CPU/GPU).
3. Identify and copy the link of a suitable [URL](https://data.pyg.org/whl/) depending on PyTorch and CUDA/CPU versions installed in the previous step. Then, run:
   - `pip install --no-index torch-scatter -f [URL]`
4. Follow [Gurobi Website](https://www.gurobi.com/) to install and setup Gurobi Optimizer.
      
### How to Use HARP
- In the `manifest` folder, The user should provide a `txt` file that holds the topology name and describes at every time step the **topology_file.json**,**set_of_pairs_file.pkl**,**traffic_matrix.pkl** file that will be read at that time step. For every timestep, a corresponding file of these three should exist in the `topologies`, `pairs`, and `traffic_matrices` folders inside a directory with the topology name. 
- For details on the data format, please check [Data Format](#data-format)
- To compute optimal values and cluterize your dataset, run:
   - ``python3 frameworks/gurobi_mlu.py --num_paths_per_pair 15 --opt_start_idx 0 --opt_end_idx 2000 --topo TopoName --framework gurobi``
   - Please refer to our paper to check the definition of a "cluster" in this context.

 - To train, run (for example):
   - ``python3 run_harp.py --topo TopoName --mode train --epochs 100 --batch_size 32 --lr 0.001 --num_paths_per_pair 8 --num_transformer_layers 2 --num_gnn_layers 3 --num_mlp1_hidden_layers 2 --num_mlp2_hidden_layers 2 --num_for_loops 3 --train_clusters 0 1 2 3 --train_start_indices 0 0 0 0 --train_end_indices 200 200 200 200 --val_clusters 4 5 --val_start_indices 0 0 --val_end_indices 90 90 --framework harp --pred 0 --dynamic 1``
   - 
   - ``python3 run_harp.py --topo abilene --mode train --epochs 100 --lr 0.007 --batch_size 32 --num_paths_per_pair 8 --num_transformer_layers 2 --num_gnn_layers 3 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 3  --train_clusters 0 --train_start_indices 0 --train_end_indices 12096 --val_clusters 0 --val_start_indices 12096 --val_end_indices 14112 --framework harp --pred 0 --dynamic 0``
   - 
   - ``python3 run_harp.py --topo kdl --mode train --epochs 100 --lr 0.007 --batch_size 8 --num_paths_per_pair 4 --num_transformer_layers 1 --num_gnn_layers 1 --num_mlp1_hidden_layers 1 --num_mlp2_hidden_layers 1 --num_for_loops 3  --train_clusters 0 --train_start_indices 0 --train_end_indices 170 --val_clusters 0 --val_start_indices 170 --val_end_indices 200 --framework harp --pred 0 --dynamic 0``

- 
- To test, run (for example):
   - ``python3 run_harp.py --topo TopoName --mode test --num_paths_per_pair 15 --num_for_loops 14  --test_cluster 6 --test_start_idx 0 --test_end_idx 150 --framework harp --pred 0 --dynamic 1``
   - ``python3 run_harp.py --topo abilene --mode test --num_paths_per_pair 8 --num_for_loops 3  --test_cluster 0 --test_start_idx 14112 --test_end_idx 16128 --framework harp --pred 0 --dynamic 0``
   - ``python3 run_harp.py --topo kdl --mode test --num_paths_per_pair 8 --num_for_loops 3  --test_cluster 0 --test_start_idx 200 --test_end_idx 278 --framework harp --pred 0 --dynamic 0``
   - Note that only one cluster is allowed per testing mode run.
- For further explanation on command line arguments, see [Command Line Arguments Explanation](#command-line-arguments-explanation)

### Working with Public Datasets (Abilene and GEANT):
- Download `AbileneTM-all.tar` from this [link](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) and decompress it (twice) inside ``prepare_abilene`` folder.
   - `cd prepare_abilene`
   - `wget https://www.cs.utexas.edu/~yzhang/research/AbileneTM/AbileneTM-all.tar`
   - `tar -xvf AbileneTM-all.tar`
   - `gunzip *.gz`
   - Then, run ``python3 prepare_abilene_harp.py``
   - This example should serve as a reference on how to prepare any dataset.
- Execute `wget --content-disposition "https://app.box.com/shared/static/shzgaxnt36org6dmu9q228kzk28numue?dl=1" -P traffic_matrices/` to download GEANT traffic matrices.
  - A preprocessed copy of the GEANT dataset in the format needed by HARP is available on this [link](https://app.box.com/s/shzgaxnt36org6dmu9q228kzk28numue)
  - Update: 09/30/2024: GEANT matrices were scaled down to have the same unit as capacities.
- Execute `wget --content-disposition "https://app.box.com/shared/static/qyq2zt160hxmmrwnt1eg792vctjmg64b?dl=1" -P traffic_matrices/` to download KDL traffic matrices.
  - A preprocessed copy of the KDL dataset in the format needed by HARP is available on this [link](https://app.box.com/s/qyq2zt160hxmmrwnt1eg792vctjmg64b).

### Working with Predicted Matrices
- By default, HARP trains over ground truth matrices.
- Running HARP with ``--pred 1`` trains it over predicted matrices rather than ground truth matrices.
- An ESM (Exponential Smoothing) predictor is provided in `traffic_matrices` directory.
  - To use it, run: `python3 esm_predictor.py TopoName`
- Provide the predicted traffic matrices using a predictor of your choice, then put them inside the `traffic_matrices` directory inside a folder named `TopoName_PredType`.
  - For example, for the GEANT dataset, original matrices will be under the `GEANT` directory whereas predicted matrices will be under the `GEANT_PredType` directory.
  - Make sure that at every time step, the predicted matrix corresponds to the ground truth matrix at that time step.
     - For example: t100.pkl in the `GEANT` and the `GEANT_PredType` folders correspond to each other.
 
 - You can specify ``--pred_type`` (default: `esm`) to indicate the predictor type if you manage multiple predicted datasets.

## Dynamic Abilene Temporal, Baseline, and Resiliency Experiments

This fork includes a dynamic Abilene pipeline used for GRATE-style temporal TE
experiments. The generated samples live in `dynamic_abilene_h6_1000_samples/`
and have six-timestep histories, time-indexed topology/path tensors, final-step
traffic matrices, and Gurobi optimal MLU labels in `sample["opt"]`.

### Current-step temporal HARP

Train the temporal model on the dynamic samples:

```bash
./train_dynamic_abilene_local_m3.sh
```

For a quick smoke test:

```bash
EPOCHS=1 TRAIN_END=20 VAL_START=20 VAL_END=30 ./train_dynamic_abilene_local_m3.sh
```

Test the trained temporal model with:

```bash
python3 run_harp.py \
  --topo dynamic_abilene \
  --mode test \
  --num_paths_per_pair 4 \
  --num_for_loops 3 \
  --test_cluster 0 \
  --framework harp \
  --pred 0 \
  --dynamic 1 \
  --dynamic_samples_dir dynamic_abilene_h6_1000_samples \
  --dynamic_test_start_idx 800 \
  --dynamic_test_end_idx 1000
```

This writes:

```text
results/dynamic_abilene/4sp/0/harp_dynamic_values_failure_id_None.txt
results/dynamic_abilene/4sp/0/harp_dynamic_stats_failure_id_None.txt
```

### Snapshot-only HARP baseline on the same data

The baseline uses the same dynamic sample files and Gurobi labels, but only the
final timestep of each sample. It does not receive temporal history.

Train:

```bash
./train_baseline_dynamic_abilene_local_m3.sh
```

Smoke test:

```bash
EPOCHS=1 TRAIN_END=20 VAL_START=20 VAL_END=30 ./train_baseline_dynamic_abilene_local_m3.sh
```

Test:

```bash
./test_baseline_dynamic_abilene_local_m3.sh
```

This writes:

```text
results/dynamic_abilene/4sp/0/harp_baseline_dynamic_values_failure_id_None.txt
results/dynamic_abilene/4sp/0/harp_baseline_dynamic_stats_failure_id_None.txt
```

Observed current-step normalized MLU on the 200-sample held-out slice
`[800, 1000)`:

```text
Temporal HARP:
  Average: 1.0267
  Median:  1.0265
  95TH:    1.0531
  Max:     1.1145

Snapshot-only HARP baseline:
  Average: 1.0837
  Median:  1.0839
  95TH:    1.1723
  Max:     1.2800
```

The temporal model wins on 194/200 paired held-out samples, with about 5.1%
mean relative improvement over the snapshot baseline. Before fixing the
temporal path encoder padding mask, the temporal checkpoint collapsed to
uniform splitting and produced average normalized MLU about 4.3037 on the same
slice; that debugging result is kept as a reference point.

### Future-failure resiliency objective

The resiliency runner trains or evaluates a model under a combined current MLU
and future-failure stress objective. It keeps the learned final-step split fixed
and evaluates single-link degradation scenarios on the final topology:

```text
combined =
  current_weight * current_norm
  + resilience_weight * (
      (1 - worst_case_weight) * expected_failure_norm
      + worst_case_weight * worst_failure_norm
    )
```

where:

- `current_norm` is current model MLU divided by the current Gurobi optimum.
- `expected_failure_norm` is the probability-weighted MLU under single-link
  degradation scenarios, normalized by the current Gurobi optimum.
- `worst_failure_norm` is the worst selected single-link degradation scenario,
  also normalized by the current Gurobi optimum.
- Scenario probabilities come from recent failure history in
  `sample["metadata"]["failed_by_t"]` plus a uniform prior.
- `failure_capacity_fraction` controls the severity. The default `0.25` means
  one scenario link keeps 25 percent of its original capacity. This is a
  differentiable stress metric, not a post-failure reoptimization LP.

Train the resilient temporal model:

```bash
./train_resilient_dynamic_abilene_local_m3.sh
```

Smoke test:

```bash
EPOCHS=1 TRAIN_END=20 VAL_START=20 VAL_END=30 ./train_resilient_dynamic_abilene_local_m3.sh
```

Useful knobs:

```bash
RESILIENCE_WEIGHT=0.5 \
WORST_CASE_WEIGHT=0.75 \
FAILURE_CAPACITY_FRACTION=0.1 \
./train_resilient_dynamic_abilene_local_m3.sh
```

Test the resilient temporal checkpoint:

```bash
./test_resilience_dynamic_abilene_local_m3.sh
```

Stress-test the current-step temporal checkpoint under the same resiliency
metric:

```bash
MODEL_TYPE=temporal \
MODEL_PATH=HARP_dynamic_dynamic_abilene_pred_False_4sp.pkl \
./test_resilience_dynamic_abilene_local_m3.sh
```

Stress-test the snapshot-only baseline checkpoint:

```bash
MODEL_TYPE=baseline \
MODEL_PATH=HARP_baseline_dynamic_abilene_pred_False_4sp.pkl \
./test_resilience_dynamic_abilene_local_m3.sh
```

Compare a trained resilient temporal checkpoint against the snapshot-only
baseline checkpoint in one run:

```bash
./compare_resilient_vs_baseline_dynamic_abilene_local_m3.sh
```

By default this expects
`HARP_resilient_temporal_dynamic_abilene_pred_False_4sp.pkl` and
`HARP_baseline_dynamic_abilene_pred_False_4sp.pkl`. Override paths or the test
slice like this:

```bash
RESILIENT_MODEL_PATH=my_resilient.pkl \
BASELINE_MODEL_PATH=HARP_baseline_dynamic_abilene_pred_False_4sp.pkl \
TEST_START=800 \
TEST_END=1000 \
./compare_resilient_vs_baseline_dynamic_abilene_local_m3.sh
```

The comparison script prints a side-by-side summary for `combined`, `current`,
`expected_failure`, and `worst_failure`, and writes detailed files under
`results/dynamic_abilene/4sp/0/resilience_compare/`.

To compare all three trained models--resilient temporal, vanilla temporal, and
snapshot-only baseline--under the same resilience evaluator:

```bash
./compare_all_dynamic_abilene_local_m3.sh
```

This expects these default checkpoints:

```text
HARP_resilient_temporal_dynamic_abilene_pred_False_4sp.pkl
HARP_dynamic_dynamic_abilene_pred_False_4sp.pkl
HARP_baseline_dynamic_abilene_pred_False_4sp.pkl
```

Override any checkpoint or the held-out slice with environment variables:

```bash
RESILIENT_MODEL_PATH=my_resilient.pkl \
TEMPORAL_MODEL_PATH=HARP_dynamic_dynamic_abilene_pred_False_4sp.pkl \
BASELINE_MODEL_PATH=HARP_baseline_dynamic_abilene_pred_False_4sp.pkl \
TEST_START=800 \
TEST_END=1000 \
./compare_all_dynamic_abilene_local_m3.sh
```

The three-way comparison writes detailed files under
`results/dynamic_abilene/4sp/0/resilience_compare_all/`.

Observed three-way resilience comparison on the 200-sample held-out slice
`[800, 1000)`:

```text
Average normalized metrics. Lower is better.

metric              resilient_temporal    vanilla_temporal    snapshot_baseline
combined                      1.848040            1.835491             1.937549
current                       1.039879            1.028735             1.083662
expected_failure              2.305775            2.339113             2.496445
worst_failure                 4.159516            4.114938             4.334649

Pairwise improvement percentages. Positive means the left model is better.

metric              resilient vs baseline    resilient vs temporal    temporal vs baseline
combined                            4.62%                   -0.68%                   5.27%
current                             4.04%                   -1.08%                   5.07%
expected_failure                    7.64%                    1.43%                   6.30%
worst_failure                       4.04%                   -1.08%                   5.07%
```

The resilient temporal checkpoint uses the same architecture as vanilla
temporal HARP; the difference is the training objective. These results suggest
that the resilience objective trades a small amount of current-step MLU for
better probability-weighted future-failure behavior: resilient temporal HARP is
best on `expected_failure`, while vanilla temporal HARP remains best on
`combined`, `current`, and `worst_failure` with the default objective weights.

The resiliency test writes four distributions per model:

```text
harp_resilient_<model_type>_dynamic_failure_id_None_combined_*.txt
harp_resilient_<model_type>_dynamic_failure_id_None_current_*.txt
harp_resilient_<model_type>_dynamic_failure_id_None_expected_failure_*.txt
harp_resilient_<model_type>_dynamic_failure_id_None_worst_failure_*.txt
```

These files are written under `results/dynamic_abilene/4sp/0/` by default.

## Reproduce Single-link Failure Experiments on Abilene and GEANT
- After training HARP model on GEANT and Abilene, run:
  - ``python3 run_failures.py --topo geant --num_paths_per_pair 8 --num_for_loops X --test_start_idx start --test_end_idx end --pred 0 --test_cluster 0``
  - ``python3 run_failures.py --topo abilene --num_paths_per_pair 8 --num_for_loops X --test_start_idx start --test_end_idx end --pred 0 --test_cluster 0``
- This will compute the optimal for each failure scenario, and then run HARP for that scenario.

### Data Format:
- **Traffic matrices**: Numpy array of shape (num_pairs, 1)
- **Pairs**: Numpy array of shape (num_pairs, 2)
- Note: the kth demand in the traffic matrix must correspond to the kth pair in the set of pairs file. This relation must be preserved for all snapshots. **We suggest sorting the hash map (pairs/keys and values/demands) before separating**.
- **Paths**: By default, HARP computes K shortest paths and automatically puts them in the correct folders and format.
   - If you wish to use your paths:
     - create a Python dictionary where the keys are the pairs and the values are a list of $K$ lists, where the inner lists are a sequence of edges.
     - For example: {(s, t): [[(s, a), (a, t)], [(s, a), (a, b), (b, t)]]}.
     - Put it inside `topologies/paths_dict` and name it: *TopoName*\_*K*\_paths_dict_cluster_*NumCluster*.pkl
        - For example: abilene_8_paths_dict_cluster_0.pkl
     - Make sure all pairs have the same number of paths (replicate if needed).

### Command Line Arguments Explanation

| Flag         | Meaning                                                                | Notes                                                                                                                                                                                                                |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| framework    | Determines the framework that solves the problem [harp, gurobi].       |                                                                                                                                                                                                                      |
| num_heads    | Number of transformer attention heads [int].                           | By default, it is equal to the number of GNN layers.                                                                                                                                                                 |
| num_for_loops| Determines HARP's Number of RAUs.                                      |                                                                                                                                                                                                                      |
| dynamic      | If your topology varies across snapshots, set it to `1`. If it is static, set it to `0`. | In our paper, the AnonNet network is dynamic.<br>GEANT, Abilene, and KDL networks are static.<br>**This CLA is useful to save GPU memory when training for a (static) topology that does not change across snapshots**.|
| dtype        | Determines the `dtype` of HARP and its data [float32, float16] corresponding to [torch.float32, torch.bfloat16]. | The default is float32.                                                                                                                                                                                              |
| checkpoint   | Enables/disables gradient checkpointing while training HARP to reduce memory footprint.         | Gradient checkpointing trades off time for memory at the level of the mini-batch. <br> Default: 0 (disabled).                                                                                                        |
| meta_learning | Turn on/off meta learning. This is used to train HARP on geant dataset for a couple of epochs before training it on the desired dataset. | This is generally useful, but specifically when the network/dataset is highly dynamic with lots of failures, where HARP might struggle to/not converge. Meta-learning significantly improves the convergence. 
| pred_type    | Label for predicted TM source                                          | Default: `esm`. |
