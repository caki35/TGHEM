# TGHEM: Topology-Guided Hard Example Mining (PyTorch)

This repository provides a PyTorch implementation of **TGHEM**, a loss function for **crowd detection** with dilated segmentation that combines:

1. **Pixel-wise segmentation loss**: Dice + BCE with **Online Hard Example Mining (OHEM)**  
2. **Topological reweighting**: a per-sample weight computed from **persistent homology** (Vietoris–Rips filtration) that measures discrepancy between **ground-truth point set** and **predicted component centroids** using **Wasserstein distance**.

The final loss is a **per-sample reweighted segmentation loss**:

$$
\begin{aligned}
\mathcal{L}_{\text{TG-HEM}}(I^{(k)}) &= w_I^{(k)} \cdot \mathcal{L}_{\text{seg}}(I^{(k)}) \\
w_I^{(k)} &= 1 + \lambda \cdot \mathcal{D}_{\text{Topo}}\bigl(C_I^{(k)}, \widehat{C}_I^{(k)}\bigr)
\end{aligned}
$$

where $\mathcal{D}_{\text{Topo}}\bigl(C_I^{(k)}, \widehat{C}_I^{(k)}\bigr)$ is the topology discrepancy for sample $k$, and $\mathcal{L}_{\text{seg}}(I^{(k)})$ is the segmentation loss computed with OHEM as

$$
\mathcal{L}_{\text{seg}}(I^{(k)}) = \text{DL}\bigl(Y_I^{(k)}, \widehat{Y}_I^{(k)}\bigr) + \sum_{q \in \texttt{HARD}_I^{(k)}} \text{BCE}(q, \widehat{q})
$$

![alt text](overview.png)
---

## Features
- **BinaryDiceLoss**: Dice loss for binary segmentation (supports `ignore_index`, `mean/sum/none` reductions).
- **SegmentationLoss**: Dice + BCE with **OHEM**, supporting:
  - confidence-threshold-based selection (`thresh` provided), or
  - top-k loss-based selection (`thresh=None`)
  - per-image OHEM when `reduction='none'`
- **TopoGuide**: topological disperency computation using persistent homology:
  - component centroid extraction from predicted binary mask
  - Vietoris–Rips filtration (Gudhi)
  - Wasserstein distance between PDs (Gudhi)
- **TGHEM**: wraps segmentation loss + topological reweighting into a single `nn.Module`.

---
## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Gudhi (with Wasserstein support)

## Input / Output

- Inputs:
    - pred_logits: torch.Tensor of shape (B, 1, H, W)
    - Raw logits (do not apply sigmoid before passing).

    - gt_mask: torch.Tensor of shape (B, H, W)
    - Binary mask in {0,1} (e.g., segmentation map derived from cell centers through slight dilation).

    - gt_dots: torch.Tensor of shape (B, H, W)
    - Dot mask where non-zero pixels indicate ground-truth point locations (e.g., cell centers).

- Output
    - Returns a scalar torch.Tensor loss.

## Citation

Please consider citing our paper if you find it useful. 
```
@article{makale,
}
```

