# TGHEM: Topology-Guided Hard Example Mining (PyTorch)

Given a mini-batch constructed from randomly cropped regions of a training image, TG-HEM computes topological features on the ground truth
and the estimated cell distributions in the regions using the Vietoris-Rips filtration, quantifies their dissimilarity via the Wasserstein distance, and increases the loss of the
regions in proportion to this topological dissimilarity during training.

This repository provides a PyTorch implementation of **TGHEM**, a loss function for **dense object detection** (e.g., crowd counting, cell segmentation) via dilated segmentation. It combines:

1.  **Pixel-wise Segmentation Loss**: Dice + BCE with **Online Hard Example Mining (OHEM)**.
2.  **Topological Reweighting**: A topological discrepancy computed between the **ground-truth point set** and **predicted component centroids** using **persistent homology** (Vietoris–Rips filtration).

The final loss is a **per-sample reweighted segmentation loss**:

$$
\begin{aligned}
\mathcal{L}_{\text{TG-HEM}}(I^{(k)}) &= w_I^{(k)} \cdot \mathcal{L}_{\text{seg}}(I^{(k)}) \\
w_I^{(k)} &= 1 + \lambda \cdot \mathcal{D}_{\text{Topo}}\bigl(C_I^{(k)}, \widehat{C}_I^{(k)}\bigr)
\end{aligned}
$$

where $\mathcal{D}_{\text{Topo}}(C_I^{(k)}, \widehat{C}_I^{(k)})$ is the topology discrepancy for sample $k$, and $\mathcal{L}_{\text{seg}}(I^{(k)})$ is the segmentation loss computed with OHEM as

$$
\mathcal{L}_{\text{seg}}(I^{(k)}) = \text{DL}\bigl(Y_I^{(k)}, \widehat{Y}_I^{(k)}\bigr) + \sum_{q \in \texttt{HARD}_I^{(k)}} \text{BCE}(q, \widehat{q})
$$

![alt text](overview.png)
---

## Features
- **BinaryDiceLoss**: Dice loss for binary segmentation (supports `ignore_index`, `mean/sum/none` reductions).
- **SegmentationLoss**: Dice + BCE with **OHEM**, supporting:
  - confidence-threshold-based hard pixel selection
  - top-k loss-based hard pixel selection
  - a safety mechanism to ensure that at least a minimum number of top-k high-loss pixels are always selected.
  - per-image OHEM reduction='none'
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
-   **Inputs**:
    -   `pred_logits`: `torch.Tensor` of shape `(B, 1, H, W)`
        -   Raw logits (do **not** apply sigmoid before passing).
    -   `gt_mask`: `torch.Tensor` of shape `(B, H, W)`
        -   Binary mask in `{0, 1}` (e.g., segmentation map derived from object centers via slight dilation).
    -   `gt_dots`: `torch.Tensor` of shape `(B, H, W)`
        -   Dot mask where non-zero pixels indicate ground-truth point locations (e.g., object centers).

-   **Output**:
    -   Returns a scalar `torch.Tensor` loss.

## Citation

Please consider citing our paper if you find it useful. 
```
@article{makale,
}
```

