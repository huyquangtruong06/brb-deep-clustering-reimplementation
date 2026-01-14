# Breaking the Reclustering Barrier in Centroid-based Deep Clustering  
**Re-implementation and Experimental Study**

---

## 📌 Overview

This repository contains our **independent re-implementation and experimental study** of the paper:

> **Breaking the Reclustering Barrier in Centroid-based Deep Clustering**  
> *Yifan Zhao, Xuhong Li, Jian Yang, Zhen Lei*  
> NeurIPS 2023  
> [[Paper]](https://arxiv.org/abs/2306.04590)

The original paper identifies and addresses the **reclustering barrier**—a fundamental limitation in centroid-based deep clustering methods where repeated reclustering leads to performance saturation or degradation.  
The authors propose **contrastive pretraining and self-labeling strategies** to break this barrier and achieve consistent performance gains.

In this project, **we re-code the core clustering baselines, learning pipelines, and evaluation protocols from scratch**, and systematically benchmark different training paradigms across multiple datasets.

---

## 🎯 Objectives

- Re-implement classical **centroid-based deep clustering algorithms** (DEC, IDEC, DCN)
- Build a **modular and reproducible experimental pipeline**
- Study the effect of:
  - Pre-trained vs. scratch training
  - Contrastive representation learning (SimCLR)
  - Self-labeling with pseudo-labels
- Validate and analyze the **reclustering barrier phenomenon** reported in the paper

---


---

## 📊 Datasets & Data Loading

We implemented **custom PyTorch DataLoaders** for the following benchmark datasets:

- **MNIST**
- **KMNIST**
- **Fashion-MNIST**
- **USPS**
- **OPTDIGITS**
- **CIFAR-10 / CIFAR-100-20 / GTSRB** (for contrastive experiments)

### Preprocessing & Augmentation

To support both clustering and contrastive learning, we implemented:

#### 🔹 Grayscale Augmentation (for MNIST-like datasets)
- Random affine transformations
  - Rotation
  - Translation
  - Scaling

#### 🔹 SimCLR-style Augmentation (for color images)
- Random resized crop
- Color jitter
- Random grayscale
- Gaussian blur
- Horizontal flip

These augmentations are critical for **contrastive representation learning**, as emphasized in the paper.

---

## 🧠 Encoder Architectures

We implemented multiple backbone architectures depending on dataset complexity:

### 1. Feed-forward Autoencoder
- Used for:
  - MNIST
  - KMNIST
  - Fashion-MNIST
  - USPS
  - OPTDIGITS
- Purpose:
  - Learning compact latent embeddings
  - Supporting reconstruction-based clustering losses

### 2. ResNet-18 Backbone
- Used for:
  - CIFAR-10
  - CIFAR-100-20
- Purpose:
  - Strong visual feature extraction
  - Contrastive pretraining (SimCLR)

---

## 🔬 Clustering Algorithms

### 1. DEC – Deep Embedded Clustering
We implemented DEC following the original formulation:

- Soft assignment using Student’s t-distribution:
  
$$
Q_{ij} = \frac{(1 + \|z_i - \mu_j\|^2)^{-1}}{\sum_k (1 + \|z_i - \mu_k\|^2)^{-1}}
$$

- **Target distribution:**

$$
P_{ij} = \frac{Q_{ij}^2 / \sum_i Q_{ij}}
{\sum_j (Q_{ij}^2 / \sum_i Q_{ij})}
$$

- **Optimization objective:**

$$
L_{clus} = KL(P \mid Q)
$$



---

### 2. IDEC – Improved DEC
IDEC extends DEC by adding a reconstruction constraint:

$$
L = L_{rec} + \gamma L_{clus}
$$

- Prevents feature drift
- Stabilizes clustering during training

---

### 3. DCN – Deep Clustering Network
We implemented the **alternating optimization scheme**:

1. Fix embeddings → update centroids
2. Fix centroids → update embeddings
3. Iterate until convergence

This iterative freeze–update strategy closely follows the original DCN design.

---

## 🔁 Contrastive Learning & Self-Labeling

### SimCLR Pretraining
For CIFAR datasets, we replaced autoencoder-based training with **contrastive learning**:

- InfoNCE loss
- Two augmented views per sample
- ResNet-18 encoder + projection head

This aligns with the paper’s claim that **contrastive representations significantly alleviate reclustering barriers**.

---

### Self-Labeling Fine-tuning

After clustering, we implemented **pseudo-label based self-labeling**:

- Assign pseudo-labels from clustering results
- Fine-tune encoder using high-confidence predictions
- Iteratively improve representation quality

---

## 📈 Evaluation Metrics

We implemented standard clustering evaluation metrics:

- **ACC** (Clustering Accuracy)
- **NMI** (Normalized Mutual Information)
- **ARI** (Adjusted Rand Index)

All metrics are computed using **best label alignment via Hungarian matching**.

---

## 🧪 Large-scale Benchmarking

We designed scripts to run experiments across all datasets under **three training paradigms**:

1. **Pre-trained**  
   (e.g., autoencoder or SimCLR pretrained)
2. **Scratch**  
   (random initialization)
3. **Contrastive**  
   (SimCLR + clustering + self-labeling)

This allows us to **empirically verify the reclustering barrier** and compare with findings reported in the paper.

---

## 📌 Key Takeaways

- Classical centroid-based methods suffer from **performance saturation under repeated reclustering**
- Contrastive pretraining significantly improves clustering robustness
- Self-labeling further enhances representation quality
- Our results qualitatively support the conclusions of:
  > *Breaking the Reclustering Barrier in Centroid-based Deep Clustering*

---

## 📚 Reference

```bibtex
@inproceedings{zhao2023breaking,
  title={Breaking the Reclustering Barrier in Centroid-based Deep Clustering},
  author={Zhao, Yifan and Li, Xuhong and Yang, Jian and Lei, Zhen},
  booktitle={NeurIPS},
  year={2023}
}
```

## 👥 Authors

This project is an independent re-implementation and study conducted by:

TRUONG QUANG HUY

NGUYEN BACH KHOA


