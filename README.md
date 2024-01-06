# Hyperspherical Prototype Node Clustering

This is the official implementation of our TMLR paper _Hyperspherical Prototype Node Clustering_.

## Prepare Environment

```bash
conda create -n hpnc
conda activate hpnc
conda install pytorch pytorch-cuda=11.8 pyg yacs pytorch-lightning tensorboardx -c pyg -c pytorch -c nvidia
```

## Reproduce the Results

To reproduce the results in Table 1, run with:
```bash
./run_all.sh
```
Then, find the results from `results/hpnc_{dec,rim}/$dataset/agg/train/best.json`

To reproduce the results in Table 2, run with:
```bash
./run_all.sh --ablation
```
