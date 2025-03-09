<div align="center">
<h2>Delving into Mapping Uncertainty for Mapless Trajectory Prediction</h2>

**Zongzheng Zhang**<sup>1,2*</sup> · **Xuchong Qiu**<sup>2*</sup> · **Boran Zhang**<sup>1</sup> · **Guantian Zheng**<sup>1</sup> · **Xunjiang Gu**<sup>4</sup> <br>
**Guoxuan Chi**<sup>1</sup> · **Huan-ang Gao**<sup>1</sup> · **Leichen Wang**<sup>2</sup> · **Ziming Liu**<sup>1</sup> · **Xinrun Li**<sup>2</sup> <br>
**Igor Gilitschenski**<sup>4</sup> · **Hongyang Li**<sup>5</sup> · [**Hang Zhao**](https://hangzhaomit.github.io/)<sup>3</sup> · [**Hao Zhao**](https://sites.google.com/view/fromandto/)<sup>1</sup>

<sup>1</sup> Institute for AI Industry Research (AIR), Tsinghua University · <sup>2</sup> Bosch Corporate Research <br>
<sup>3</sup> Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University <br>
<sup>4</sup> University of Toronto · <sup>5</sup> The University of Hong Kong <br>
<sub>(* indicates equal contribution)</sub>
</div>

This repository contains the implementation of our ongoing research on mapping uncertainty for mapless trajectory prediction, which we plan to submit to IROS 2025.
We propose a lightweight, self-supervised approach that enhances the synergy between online mapping and trajectory prediction, providing interpretability on when and where map uncertainty is beneficial. Additionally, we introduce a covariance-based uncertainty modeling method that better aligns with road geometry. Extensive ablation studies show that our method outperforms previous integration strategies, achieving up to 23.6% improvement in mapless trajectory prediction on the nuScenes dataset.

![pipeline](assets/overview.png)
Our framework consists of a mapping module based on Covariance Map Uncertainty estimation and a trajectory prediction module with Proprioceptive Scenario Gating, enabling selective integration of upstream uncertainty with vehicle kinematics. We utilize online mapping methods, including MapTR, MapTRv2, MapTRv2-Centerline, and StreamMapNet, to regress map element vertices, where each vertex’s uncertainty is modeled using a 2D Gaussian distribution. This uncertainty information is then leveraged to refine trajectory predictions.

For trajectory prediction, we employ two representative models: HiVT, a Transformer-based approach, and DenseTNT, a GNN-based approach. During context encoding, map vertex coordinates (mean μ(i)) are first processed via an MLP. To incorporate uncertainty, we concatenate the mean μ(i) and covariance matrix Σ(i) into a unified representation, which is further processed through an MLP. This process runs in two parallel streams—one incorporating uncertainty and one without. The Proprioceptive Scenario Gating mechanism dynamically fuses the outputs from both streams using an MLP that assigns adaptive weights, enabling the model to generate more accurate and robust trajectory predictions.

To evaluate the impact of our uncertainty modeling strategy on downstream trajectory prediction, we conduct experiments across multiple combinations of online map estimation(MapTR, MapTRv2, MapTRv2-Centerline and StreamMapNet) and prediction methods(HiVT and DenseTNT). Our approach consistently outperforms existing methods, achieving state-of-the-art results across all evaluated settings. With HiVT, our method improves minADE and minFDE by over 6% on MapTR and MapTRv2, while also achieving a significant 13.6% reduction in MR on MapTRv2. For DenseTNT, the largest gains are observed on MapTRv2-Centerline, with minADE, minFDE, and MR improving by 19.4%, 10.8%, and 23.6%, respectively. These results demonstrate the effectiveness of our uncertainty modeling in enhancing trajectory prediction accuracy.(compared with base???)

| Prediction Method |  |HiVT|  |  |DenseTNT|  |
|------------------------|---------|---------|---------|---------|---------|---------|
| Online HD Map Method | minADE↓ | minFDE↓ | MR↓ | minADE↓ | minFDE↓ | MR↓ |
| **MapTR [base]** | 0.4015 | 0.8404 | 0.0960 | 1.1228 | 2.2151 | 0.3726 |
| **MapTR [base] + Unc [Gu]** | 0.3910 | 0.8049 | 0.0818 | 1.1946 | 2.2666 | 0.3848 |
| **MapTR [base] + CovMat [Ours]** | 0.3672 | 0.7395 | 0.0756 | 1.0856 | 2.0969 | 0.3728 |
| **MapTR [bev]** | 0.3617 | 0.7401 | 0.0720 | 0.7608 | 1.4700 | 0.2593 |
| **MapTR [bev] + CovMat [Ours]** | 0.3498 | 0.7021 | 0.0651 | - | - | - |
| **MapTR2 [base]** | 0.4017 | 0.8406 | 0.0959 | 1.3262 | 2.5687 | 0.4301 |
| **MapTRv2 [base] + Unc [Ours]** | 0.3913 | 0.8054 | 0.0819 | 1.3256 | 2.6390 | 0.4435 |
| **MapTRv2 [base] + CovMat [Ours]** | 0.3670 | 0.7538 | 0.0708 | 1.1585 | 2.4566 | 0.3891 |
| **MapTRv2 [bev]** | 0.3844 | 0.7848 | 0.0741 | 1.1232 | 2.3000 | 0.4025 |
| **MapTR2 [bev] + CovMat [Ours]** | 0.3423 | 0.7285 | 0.0667 | - | - | - |
| **MapTRv2-CL [base]** | 0.3789 | 0.7859 | 0.0865 | 0.8333 | 1.4752 | 0.1719 |
| **MapTRv2-CL [base] + Unc [Ours]** | 0.3674 | 0.7418 | 0.0739 | 0.9666 | 1.6439 | 0.2082 |
| **MapTRv2-CL [base] + CovMat [Ours]** | 0.3659 | 0.7404 | 0.0721 | 0.7787 | 1.4662 | 0.1590 |
| **MapTRv2-CL [bev]** | 0.3652 | 0.7323 | 0.0710 | 0.7630 | 1.3609 | 0.1576 |
| **MapTRv2-CL [bev] + CovMat [Ours]** | 0.3496 | 0.7096 | 0.6794 | - | - | - |
| **StreamMapNet [base]** | 0.3963 | 0.8223 | 0.0923 | 1.0639 | 2.1430 | 0.3412 |
| **StreamMapNet [base] + Unc [Ours]** | 0.3899 | 0.8101 | 0.0861 | 1.0902 | 2.1412 | 0.3261 |
| **StreamMapNet [base] + CovMat [Ours]** | 0.3870 | 0.7995 | 0.0834 | 0.9675 | 1.6883 | 0.2628 |
| **StreamMapNet [bev]** | 0.3800 | 0.7709 | 0.0746 | 0.7377 | 1.3661 | 0.1987 |
| **StreamMapNet [bev] + CovMat [Ours]** | - | - | - | - | - | - |

This is our demo video:

https://github.com/user-attachments/assets/c21ff2ed-4b79-4e5b-b939-9f0c2b32c1e2
## Getting Started
- [Environment Setup](docs/env.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Mapping Train and Eval](docs/map.md)
- [Merge Map and Trajectory Dataset](docs/adaptor.md)
- [Trajectory Train and Eval](docs/trj.md)
- [Visualization](docs/visualization.md)

## Dataset

All the trajectory prediction data（for `MapTR`, `StreamMapNet`, `MapTRv2` and `MapTRv2 CL`）can be generated using our future checkpoints, with a total size of approximately 600GB.
Dataset Structure is as follows:
```
DelvingUncPrediction
├── trj_data/
│   ├── maptr/
│   |   ├── train/
│   |   |   ├── data/
│   |   |   |   ├── scene-{scene_id}.pkl
│   |   ├── val/
│   ├── maptrv2/
│   ├── maptrv2_CL/
│   ├── stream/
```

## Catalog

- [x] Code release
  - [x] MapTR
  - [x] MapTRv2
  - [x] StreamMapNet
  - [x] HiVT
  - [x] DenseTNT
- [x] Visualization Code
- [x] Untested version released + Instructions
- [x] Initialization




## License

This repository is licensed under [Apache 2.0](LICENSE).
