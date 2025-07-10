# SimCortex: Collision-free Simultaneous Cortical Surface Reconstruction

<p align="center">
  <img src="figure/pipeline_overview.jpg" width="60%" alt="SimCortex Pipeline"/>
</p>

This is the **official PyTorch implementation** of our paper:

> [**SimCortex: Collision-free Simultaneous Cortical Surface Reconstruction**](https://arxiv.org/abs/2507.06955)  
> Kaveh Moradkhani, R. J. Rushmore, Sylvain Bouix  
> ShapeMI MICCAI (2025) 

---

## Pipeline Overview

SimCortex proceeds in three consecutive stages:

1. **Multi-Class Segmentation:**
   A 3D U-Net segments each T1-weighted MRI into nine tissue classes.

2. **Subject-Specific Initial Surface Generation:**
Binary masks are created from tissue labels, then signed-distance fields are computed and corrected to generate collision-free initial cortical surfaces.

3. **Simultaneous Multiscale Diffeomorphic Surface Reconstruction:**
Four initial surfaces are jointly refined using multiscale velocity fields to produce smooth, topology-preserving, and collision-free cortical meshes.

---

## Installation
## Dataset
## Evaluation
## Training

## Citation
