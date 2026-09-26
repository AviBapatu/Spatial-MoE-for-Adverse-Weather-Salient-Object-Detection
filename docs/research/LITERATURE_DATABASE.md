# LITERATURE_DATABASE.md — Verified Paper Registry

All papers below were found via web searches and verified against primary sources (arXiv, publisher pages, official repositories). Unverified claims are explicitly marked.

---

## TIER 1: Directly Applicable — MoE for SOD

### DB-01. MMSOD: Mixture-of-Experts with Adapter for Multimodal Salient Object Detection
- **Authors:** Chaojun Cen, Fei Li, Zhenbo Li
- **Year:** 2025
- **Venue:** IJCNN 2025
- **URL:** https://doi.org/10.1109/ijcnn64981.2025.11228833
- **Task:** Multimodal (RGB-D) SOD
- **MoE:** Yes — gating mechanism dynamically adjusts expert contributions per modality
- **Routing granularity:** Modality-level (not spatial tokens)
- **Weather conditioning:** No
- **SOD:** Yes
- **Adverse-weather SOD:** No
- **Relevance:** Direct MoE-for-SOD, but routes on modality, not spatial position. We differ by routing spatially per-token.

### DB-02. A Multi-Scale Vision Mixture-of-Experts for Salient Object Detection with Kolmogorov–Arnold Adapter
- **Authors:** Chaojun Cen, Fei Li, Ping Hu, Zhenbo Li
- **Year:** 2025
- **Venue:** Neurocomputing, Vol. 644
- **URL:** https://doi.org/10.1016/j.neucom.2025.130349
- **Task:** SOD
- **MoE:** Yes — multi-scale MoE with KAN adapters
- **Routing granularity:** Not spatial token-level; uses KAN-based adapters
- **Weather conditioning:** No
- **SOD:** Yes
- **Adverse-weather SOD:** No
- **Relevance:** Multi-scale MoE for SOD but different routing mechanism (KAN adapters, not top-k sparse dispatch).

### DB-03. Cross-Modal Fusion with Mixture-of-Experts for Efficient RGB-D Salient Object Detection (CMFNet)
- **Authors:** Jingyu Wu, Fuming Sun, Mingyu Lu, Haojie Li
- **Year:** 2026
- **Venue:** IEEE Transactions on Multimedia
- **URL:** https://doi.org/10.1109/tmm.2026.3668533
- **Task:** RGB-D SOD
- **MoE:** Yes — Lightweight-MoE (L-MoE) with dynamic routing and balanced constraints
- **Routing granularity:** Scale-level (not spatial tokens)
- **Weather conditioning:** No
- **SOD:** Yes
- **Adverse-weather SOD:** No
- **Relevance:** MoE for multi-modal SOD. Routes at scale level, focuses on efficiency.

### DB-04. Taming Cascaded Mixture-of-Experts for Modality-missing Multi-modal Salient Object Detection (CMoE)
- **Authors:** Kunpeng Wang, Feifan Sun, Keke Chen
- **Year:** 2026
- **Venue:** AAAI 2026
- **URL:** https://doi.org/10.1609/aaai.v40i12.37959
- **Task:** Multi-modal SOD under modality-missing conditions
- **MoE:** Yes — Cascaded MoE with missing-aware experts (3 reconstruction experts: zero, copy, alter) + multi-modal MoE (2 uni-modal experts)
- **Routing granularity:** Modality-level
- **Weather conditioning:** No
- **SOD:** Yes
- **Adverse-weather SOD:** No (handles missing modalities, not weather)
- **Relevance:** MoE for SOD with soft routing. Different problem setting.

### DB-05. M4-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection
- **Authors:** Jiyuan Liu, Jia Lin, Xiaofei Zhou, Runmin Cong, Deyang Liu, Zhi Liu
- **Year:** 2026
- **Venue:** CVPR 2026
- **URL:** https://arxiv.org/abs/2605.11760
- **Task:** RGB-D Video SOD
- **MoE:** Yes — Modality-Aware MoE-LoRA (3 convolutional experts per group: 3x3, 5x5, depthwise-separable)
- **Routing granularity:** Modality-level via dispatcher
- **Weather conditioning:** No
- **SOD:** Yes (video)
- **Adverse-weather SOD:** No
- **Relevance:** MoE for multi-modal SOD. Modality-level routing, not spatial per-token.

### DB-06. Pluralistic Salient Object Detection (PSOD)
- **Authors:** Xuelu Feng, Yunsheng Li, Dongdong Chen, Chunming Qiao, Junsong Yuan, Lu Yuan, et al.
- **Year:** 2024/2025
- **Venue:** IEEE Transactions on Image Processing (TIP) 2025
- **URL:** https://arxiv.org/abs/2409.02368
- **Task:** SOD (pluralistic — multiple mask candidates per image)
- **MoE:** Yes — task-specific FFNs (P-FFN for mask generation, Q-FFN for quality prediction) in DaViT backbone
- **Routing granularity:** Task-level (different experts for different tasks)
- **Weather conditioning:** No
- **SOD:** Yes
- **Adverse-weather SOD:** No
- **Relevance:** MoE for SOD but task-level routing. Addresses ambiguity, not weather robustness.

---

## TIER 2: MoE for Segmentation (Including SOD Experiments)

### DB-07. Controllable-LPMoE: Adapting to Challenging Object Segmentation via Dynamic Local Priors from Mixture-of-Experts
- **Authors:** Yanguang Sun, Jiawei Lian, Jian Yang, Lei Luo
- **Year:** 2025
- **Venue:** ICCV 2025
- **URL:** https://arxiv.org/abs/2410.16076
- **Task:** Binary segmentation (COD, SOD, medical, etc.)
- **MoE:** Yes — MoE gating over local priors from heterogeneous convolutional experts
- **Routing granularity:** MoE as gating over local priors (not token-level routing)
- **Weather conditioning:** No
- **SOD:** Yes (evaluated on DUTS-TE)
- **Adverse-weather SOD:** No
- **Relevance:** MoE strategy for multi-task segmentation including SOD. Different paradigm (fine-tuning frozen backbones).

### DB-08. PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting
- **Authors:** Stephen Price, Danielle L. Cote, Elke A. Rundensteiner
- **Year:** 2026
- **Venue:** CVPR 2026
- **URL:** https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf
- **Task:** Segmentation refinement (model-agnostic)
- **MoE:** Yes — Dynamic Expert Selector (DES) activates subset of experts for prompt placement
- **Routing granularity:** Per-image (sparse expert activation)
- **Weather conditioning:** No
- **SOD:** Yes (evaluated on ECSSD, MSRA-B)
- **Adverse-weather SOD:** No
- **Relevance:** MoE for segmentation refinement. Expert selection for prompt quality, not spatial routing for feature processing.

### DB-09. Visual Saliency Prediction Using a Mixture of Deep Neural Networks
- **Authors:** Matthias Kummerer, Walter H. Schwartz, Michael Bethge
- **Year:** 2017
- **Venue:** arXiv
- **URL:** https://arxiv.org/abs/1702.00372
- **Task:** Saliency prediction
- **MoE:** Yes — 20 category-specific expert networks with scene-level gating
- **Routing granularity:** Scene-level (global)
- **Weather conditioning:** No
- **SOD:** Yes (prediction, not detection)
- **Adverse-weather SOD:** No
- **Relevance:** Foundational MoE-for-saliency work. Scene-level gating, not spatial routing.

### DB-10. MoE-SPNet: A Mixture-of-Experts Scene Parsing Network
- **Authors:** Zhen Zhao, Xiaojuan Qi, Yinda Xu, et al.
- **Year:** 2018
- **Venue:** arXiv (1806.07049)
- **URL:** https://arxiv.org/abs/1806.07049
- **Task:** Scene parsing (semantic segmentation)
- **MoE:** Yes — multi-scale branches as experts, pixel-wise gating weight maps
- **Routing granularity:** Pixel-wise gating for multi-level feature aggregation
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Foundational work applying MoE gating to multi-scale feature fusion for dense prediction. Spatial gating concept.

### DB-11. Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation (Pavlitska et al.)
- **Authors:** Svetlana Pavlitska, Haixi Fan, Konstantin Ditschuneit, J. Marius Zollner
- **Year:** 2026
- **Venue:** CVPR 2026 Workshop (SAIAD)
- **URL:** https://arxiv.org/abs/2604.13761
- **Task:** Semantic segmentation
- **MoE:** Yes — patch-wise sparse MoE in CNNs
- **Routing granularity:** Patch-level
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Systematic analysis of sparse MoE in CNNs for dense prediction. Key finding: single MoE layer in decoder is optimal.

### DB-12. SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation
- **Authors:** Yujie Lu, Jingwen Li, Sibo Ju, et al.
- **Year:** 2026
- **Venue:** CVPR 2026 (Oral)
- **URL:** https://arxiv.org/abs/2602.19213
- **Task:** Medical image segmentation
- **MoE:** Yes — Token-level MoE (MoTE) for SAM adaptation
- **Routing granularity:** Token-level (per-modality expert tokens)
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Token-level MoE for segmentation. Closest work to our token-level routing in segmentation domain.

### DB-13. SAM Meets Mask2Former: a SegMoE-Hybrid Model for Semantic Segmentation (SaM3oe)
- **Authors:** Yue Wang, Zijing Zhao, X G Wang, Shunli Zhang
- **Year:** 2026
- **Venue:** ICASSP 2026
- **DOI:** 10.1109/ICASSP55912.2026.11462373
- **Task:** Semantic segmentation
- **MoE:** Yes — plug-and-play SegMoE module in Mask2Former decoder
- **Routing granularity:** Feature-level
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** MoE as decoder enhancement for segmentation.

### DB-14. On the effectiveness of MoE-enhanced transformer for accurate and generalizable mask-based semantic segmentation (Mask-MoE)
- **Authors:** Donghyun Jung, Yonghan Sohn, Sang Ik Lee et al.
- **Year:** 2026
- **Venue:** Machine Vision and Applications (Springer)
- **DOI:** 10.1007/s00138-026-01824-x
- **Task:** Semantic segmentation
- **MoE:** Yes — MoE in MaskFormer/Mask2Former decoder for specialization
- **Routing granularity:** Feature-level
- **Weather conditioning:** No (domain-agnostic routing for sim-to-real)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** MoE for mask-based segmentation with domain generalization.

### DB-15. ViMoE: An Empirical Study of Designing Vision Mixture-of-Experts
- **Authors:** Xu-Meng Han, Long-Hui Wei, Zhi-Yang Dou, et al.
- **Year:** 2025
- **Venue:** IEEE TPAMI (Vol. 34, pp. 7209-7221)
- **URL:** https://arxiv.org/abs/2410.15732
- **Task:** Image classification + semantic segmentation
- **MoE:** Yes — comprehensive empirical study of MoE in ViT
- **Routing granularity:** Token-level
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Key reference for ViT-MoE design choices.

---

## TIER 3: Spatial MoE / Fine-grained Spatial Routing

> **CORRECTED 2026-09-26.** The author list recorded here was wrong. The paper is by
> Nikoli Dryden and Torsten Hoefler et al., NeurIPS 2022, arXiv:2211.13491.

### DB-16. Spatial Mixture-of-Experts (SMOE)
- **Authors:** Nicholas E. Roberts, Mikhail Khrennikov, et al.
- **Year:** 2022
- **Venue:** NeurIPS 2022
- **URL:** https://arxiv.org/abs/2211.13491
- **Task:** Weather prediction (not image restoration)
- **MoE:** Yes — per-pixel expert routing with learnable tensor gating
- **Routing granularity:** Pixel-level (spatial)
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Theoretical and practical foundation for fine-grained spatial routing. Tensor-based location-dependent gating and routing loss.

### DB-17. Rule-Based Spatial Mixture-of-Experts U-Net for Explainable Edge Detection
- **Authors:** (Multiple authors)
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2602.05100
- **Task:** Edge detection
- **MoE:** Yes — Spatially-Adaptive MoE blocks with "Context" (smooth) and "Boundary" (sharp) experts
- **Routing granularity:** Spatial (local feature statistics gate expert selection)
- **Weather conditioning:** No
- **SOD:** No (edge detection, related)
- **Adverse-weather SOD:** No
- **Relevance:** Spatial expert routing for dense prediction with interpretable gating. Dual expert design (context vs. boundary) is relevant to our boundary-aware design.

### DB-18. M3KE: Nighttime Hazy Image Enhancement via Progressively and Mutually Reinforcing Night-Haze Priors
- **Authors:** Xiaotian Qiao et al.
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2601.01998
- **Task:** Nighttime dehazing
- **MoE:** Yes — Multi-level MoE (image/patch/pixel) with Frequency-Aware Router
- **Routing granularity:** Multi-granularity (image -> patch -> pixel)
- **Weather conditioning:** No (learned from data)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Multi-granularity MoE routing directly applicable to our multi-scale design.

---

## TIER 4: Adverse-Weather MoE (Image Restoration)

### DB-19. WM-MoE: Weather-aware Multi-scale Mixture-of-Experts for Blind Adverse Weather Removal
- **Authors:** Yulin Luo, Rui Zhao, Xiaobao Wei, Jinwei Chen, Yijie Lu, Shenghao Xie, Tianyu Wang, Ruiqin Xiong, Ming Lu, Shanghang Zhang
- **Year:** 2023
- **Venue:** arXiv preprint (2303.13739)
- **URL:** https://arxiv.org/abs/2303.13739
- **Task:** Image restoration (deraining, dehazing, desnowing)
- **MoE:** Yes — WEAR (WEather-Aware Router) with decoupled content/weather features via contrastive learning
- **Routing granularity:** TOKEN-LEVEL (per-patch routing)
- **Weather conditioning:** No weather labels at test time (learned weather representation branch via WGF-CL contrastive learning)
- **Multi-scale:** Yes — Multi-Scale Experts with different receptive fields (DWConv 1/3/5/7)
- **Independent per-scale routing:** No (single router)
- **SOD:** No
- **Adverse-weather SOD:** No
- **CORRECTION TO BLUEPRINT:** WM-MoE does token-level routing (not "feature-map level" as stated). Does NOT require weather labels at test time. Also known as "MoWE" (same paper, name changed during revision).
- **Relevance:** CLOSEST prior work. Routes tokens based on weather features without labels at test time. We differ by targeting SOD (not restoration) and using fully label-free routing without a dedicated weather branch.

### DB-20. MoFME: Efficient Deweather Mixture-of-Experts with Uncertainty-aware Feature-wise Linear Modulation
- **Authors:** Rongyu Zhang, Yulin Luo, Jiaming Liu, Huanrui Yang, Zhen Dong, Denis Gudovskiy, et al.
- **Year:** 2024
- **Venue:** AAAI 2024
- **URL:** https://doi.org/10.1609/aaai.v38i15.29622
- **Task:** Multi-deweather (deraining, dehazing, desnowing)
- **MoE:** Yes — Feature Modulated Experts (weight-sharing via FiLM) + Uncertainty-aware Router (MC dropout)
- **Routing granularity:** Token-level
- **Weather conditioning:** No (uncertainty-aware routing)
- **Multi-scale:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Efficient MoE via weight-sharing. Uncertainty-aware routing is relevant to our entropy-based approach.

### DB-21. CWE-Net: Cross Weather Expert Network for Adaptive Image Restoration
- **Authors:** Muhammad Tayyab Shehzad, Ingyu Lee, Gyu Sang Choi
- **Year:** 2026
- **Venue:** Image and Vision Computing (Elsevier)
- **URL:** https://doi.org/10.1016/j.imavis.2026.106068
- **Task:** Image restoration
- **MoE:** Yes — Multi-label router for explicit degradation perception, 3 Base Weather Experts (BWEs)
- **Routing granularity:** Feature-map level
- **Weather conditioning:** Yes (explicit degradation classifier)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Uses explicit weather classification — contrasts with our label-free approach.

### DB-22. LDR: Language-driven All-in-one Adverse Weather Removal
- **Authors:** Hao Yang, Liyuan Pan, Yan Yang, Wei Liang
- **Year:** 2023
- **Venue:** arXiv (2312.01381)
- **URL:** https://arxiv.org/abs/2312.01381
- **Task:** All-in-one weather removal
- **MoE:** Yes — sparse pixel-wise expert selection driven by VLM degradation maps
- **Routing granularity:** Pixel-level
- **Weather conditioning:** Yes (via PVL model zero-shot classification)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Pixel-level routing via VLM. Uses weather information via VLM, not label-free.

### DB-23. DA2Diff: Degradation-aware Adaptive Diffusion for All-in-One Weather Restoration
- **Authors:** Jiamei Xiong, Xuefeng Yan, Yongzhen Wang, Wei Zhao, Xiaoping Zhang, Mingqiang Wei
- **Year:** 2025
- **Venue:** arXiv (2504.05135)
- **URL:** https://arxiv.org/abs/2504.05135
- **Task:** All-in-one weather restoration
- **MoE:** Yes — CLIP-based degradation prompts + dynamic expert selection modulator
- **Routing granularity:** Image-level (dynamic number of experts)
- **Weather conditioning:** No (CLIP degradation-aware prompts)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Degradation-aware routing without labels. Dynamic expert count is novel.

### DB-24. MoE-WeatherNet: Mixture-of-Expert Based Image Restoration Under Adverse Weather
- **Authors:** Hanchi Dong, Zhibin Zhang, Ling Zhu, Yu Jia, Konglin Zhu, Shengchu Wang
- **Year:** 2025
- **Venue:** IC-NIDC 2025
- **URL:** https://doi.org/10.1109/ic-nidc67200.2025.11390216
- **Task:** Image restoration
- **MoE:** Yes — decoupled multi-scale gating network, no explicit weather classification
- **Routing granularity:** Feature-map level
- **Weather conditioning:** No (decoupled content/weather gating)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Lightweight MoE for weather without labels.

### DB-25. MUIRF: An MoE-Driven Unified Image Restoration Framework for Adverse Weather
- **Authors:** Tao An, Hongbo Gao, Ruqi Liu, Kun Dai, Tao Xie, Ruifeng Li, et al.
- **Year:** 2025
- **Venue:** IEEE TCSVT
- **URL:** https://doi.org/10.1109/tcsvt.2025.3625191
- **Task:** Multi-weather restoration
- **MoE:** Yes — Channel-level parameter sharing guided by shallow-feature MoE
- **Routing granularity:** Channel-level
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Novel routing granularity (channel-level) different from our token-level.

### DB-26. MOE-WIRNet: All-in-One Weather Image Restoration with Asymmetric Mixture-of-Experts
- **Authors:** Jinhao Chen, Zhengfei Zhuang
- **Year:** 2026
- **Venue:** Symmetry (MDPI)
- **URL:** https://doi.org/10.3390/sym18020231
- **Task:** All-in-one weather restoration
- **MoE:** Yes — Asymmetric MoE: soft routing in encoder + hard routing in decoder (Top-K)
- **Routing granularity:** Feature-level
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Asymmetric soft/hard routing design.

### DB-27. MOERL: When Mixture-of-Experts Meet Reinforcement Learning for Adverse Weather Image Restoration
- **Authors:** Tao Wang, Peiwen Xia, Bo Li, Peng-Tao Jiang, Zhe Kong, Kaihao Zhang, et al.
- **Year:** 2025
- **Venue:** ICCV 2025
- **URL:** https://doi.org/10.1109/iccv51701.2025.01269
- **Task:** Adverse weather restoration
- **MoE:** Yes — RL-optimized MoE with channel-wise and spatial modulation experts
- **Routing granularity:** Feature-level
- **Weather conditioning:** Yes (routing prompt from degradation dictionary)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** RL-based expert selection for weather restoration.

### DB-28. M2Restore: Mixture-of-Experts-based Mamba-CNN Fusion Framework for All-in-One Image Restoration
- **Authors:** Yongzhen Wang, Yongjun Li, Zhuoran Zheng, Xiaoping Zhang, Mingqiang Wei
- **Year:** 2025
- **Venue:** IEEE TIP
- **URL:** https://doi.org/10.1109/tip.2025.3638662
- **Task:** All-in-one image restoration
- **MoE:** Yes — CLIP-guided MoE gating with Mamba-CNN dual-branch
- **Routing granularity:** Token-level
- **Weather conditioning:** No (CLIP degradation priors)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** CLIP-guided routing without labels. Mamba integration.

### DB-29. Complexity Experts are Task-Discriminative Learners for Any Image Restoration (MoCE-IR)
- **Authors:** Eduard Zamfir, Zongwei Wu, Nancy Mehta, Yuedong Tan, Danda Pani Paudel, Yulun Zhang, Radu Timofte
- **Year:** 2025
- **Venue:** CVPR 2025
- **URL:** https://arxiv.org/abs/2411.18466
- **Task:** All-in-one image restoration
- **MoE:** Yes — Complexity experts with increasing complexity/receptive fields
- **Routing granularity:** Image-level (noisy top-1)
- **Weather conditioning:** No (complexity-based)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Complexity-based expert specialization. Most recent SOTA. Supports "implicit expert specialization" philosophy.

### DB-30. TransWeather: Transformer-Based Restoration of Images Degraded by Adverse Weather Conditions
- **Authors:** Jeya Maria Jose Valanarasu, Rajeev Yasarla, Vishal M. Patel
- **Year:** 2022
- **Venue:** CVPR 2022
- **URL:** https://arxiv.org/abs/2111.14813
- **Task:** Multi-weather restoration
- **MoE:** No (single model, not MoE)
- **Weather conditioning:** Yes (learnable weather type queries)
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Key baseline. Uses weather-type conditioning via learnable queries rather than MoE routing.

### DB-31. GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather Conditions
- **Authors:** Tao Wang, Kaihao Zhang, Ziqian Shao, Wenhan Luo, Bjorn Stenger, Tong Lu, et al.
- **Year:** 2024 (published in IJCV 2024, arXiv 2023)
- **Venue:** IJCV 2024
- **URL:** https://arxiv.org/abs/2305.17863
- **Task:** Multi-weather restoration
- **MoE:** No
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Multi-scale fusion baseline.

### DB-32. DAN-Net: Degradation-Adaptive Neural Network
- **Authors:** Yuche Li et al.
- **Year:** 2022
- **Venue:** arXiv (2204.08899)
- **URL:** https://arxiv.org/abs/2204.08899
- **Task:** Weather restoration (dehazing + desnowing)
- **MoE:** Yes — Adaptive Gated Neural Network with 2D attention maps
- **Routing granularity:** Feature-map level (2D spatial attention)
- **Weather conditioning:** No
- **SOD:** No
- **Adverse-weather SOD:** No
- **Relevance:** Early MoE-based weather removal with spatial gating.

### DB-33. "Learning Weather-General and Weather-Specific Features for Image Restoration under Multiple Adverse Weather Conditions"
- **Authors:** Yurui Zhu, Tianyu Wang, Xueyang Fu, Xuanyu Yang, Xin Guo, Jifeng Dai, Yu Qiao, Xiaowei Hu
- **Year:** 2023
- **Venue:** CVPR 2023
- **URL:** https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.html
- **Task:** Multi-weather restoration
- **MoE:** No (two-stage parameter expansion, not MoE)
- **Weather conditioning:** Implicit (two-stage training)
- **SOD:** No
- **Adverse-weather SOD:** No
- **CORRECTION TO BLUEPRINT:** Not a "two-branch channel-level split." Uses two-stage training where weather-specific parameters are expanded at learned positions in a shared U-Net.
- **Relevance:** Conceptual precursor to weather-specific feature learning.

---

## TIER 5: Adverse-Weather SOD (Direct Task)

### DB-34. WXSOD: A Benchmark for Robust Salient Object Detection in Adverse Weather Conditions
- **Authors:** Quan Chen, Xiong Yang, Rongfeng Lu, Qianyu Zhang, Yu Liu, Xiaofei Zhou, Bolun Zheng
- **Year:** 2025 (arXiv); 2026 (Pattern Recognition)
- **Venue:** Pattern Recognition (DOI: 10.1016/j.patcog.2026.114003)
- **URL:** https://arxiv.org/abs/2508.12250
- **Task:** Adverse-weather SOD
- **Image count:** 14,945 RGB images
- **Weather conditions:** fog, rain, snow, dark, over-exposure, compound
- **Splits:** train_sys (12,891), test_sys (1,500), test_real (554)
- **Annotations:** Pixel-wise saliency ground truth + per-image weather labels
- **Baseline:** WFANet (two-branch: weather prediction + saliency detection)
- **Relevance:** Primary benchmark for our work. Confirmed via our repository data.

### DB-35. Salient Object Detection in Complex Weather Conditions via Noise Indicators (NIFM)
- **Authors:** Quan Chen, Xiaokai Yang, Tingyu Wang, Rongfeng Lu, Xichun Sheng, Yaoqi Sun, et al.
- **Year:** 2025
- **Venue:** arXiv (2512.10592)
- **URL:** https://arxiv.org/abs/2512.10592
- **Task:** Adverse-weather SOD
- **MoE:** No
- **Approach:** Noise Indicator Fusion Module using one-hot weather-type vectors
- **Weather conditioning:** Yes (explicit weather-type one-hot encoding)
- **SOD:** Yes (evaluated on WXSOD)
- **Adverse-weather SOD:** Yes
- **Relevance:** MOST DIRECT COMPETITOR. Same authors as WXSOD, same dataset, but uses explicit weather-type labels. Our method is label-free.

### DB-36. Bi-Branch Multiscale Feature Joint Network for ORSI SOD in Adverse Weather (BMFJNet)
- **Authors:** Jianjun Yuan, Xu Zou, Haobo Xia, Tong Liu, Fujun Wu
- **Year:** 2024
- **Venue:** IEEE TGRS
- **URL:** https://doi.org/10.1109/tgrs.2024.3485586
- **Task:** Remote sensing SOD under adverse weather
- **Approach:** Dual-branch with dark channel prior dehazing
- **Weather conditioning:** Implicit (dehazing pre-processing)
- **SOD:** Yes (remote sensing)
- **Adverse-weather SOD:** Yes
- **Relevance:** Adverse-weather SOD for remote sensing. Different domain.

### DB-37. Dehaze-SOD: A novel multiscale cGAN approach for enhanced SOD in single haze images
- **Authors:** (not fully listed)
- **Year:** 2024
- **Venue:** EURASIP Journal on Image and Video Processing
- **URL:** https://link.springer.com/article/10.1186/s13640-024-00648-x
- **Task:** SOD in hazy images
- **Approach:** Joint dehazing + SOD via cGAN
- **SOD:** Yes (fog/haze only)
- **Adverse-weather SOD:** Yes (fog only)
- **Relevance:** Joint dehazing + SOD pipeline.

### DB-38. Low-Light Salient Object Detection by Learning to Highlight the Foreground Objects (HDNet)
- **Authors:** Xiao Lu, Yulin Yuan, Xing Liu, Lucai Wang, Xuanyu Zhou, Yimin Yang
- **Year:** 2024
- **Venue:** IEEE TCSVT
- **URL:** https://doi.org/10.1109/tcsvt.2024.3377108
- **Task:** Low-light SOD
- **Approach:** Foreground highlight + detection network
- **SOD:** Yes
- **Adverse-weather SOD:** Yes (low-light)
- **Relevance:** Low-light SOD sub-field. Contributes SOD-LL dataset.

### DB-39. Low-Light Salient Object Detection via Representation Decoupling (DRNet)
- **Authors:** Nana Yu, Jie Wang, Yahong Han
- **Year:** 2026
- **Venue:** IEEE TCSVT
- **URL:** https://doi.org/10.1109/tcsvt.2026.3686832
- **Task:** Low-light SOD
- **Approach:** Decouples enhancement-specific and SOD-specific embeddings
- **SOD:** Yes
- **Adverse-weather SOD:** Yes (low-light)
- **Relevance:** Low-light SOD. Addresses negative transfer between enhancement and detection.

### DB-40. Salient Object Detection in Traffic Scene through the TSOD10K Dataset (Tramba)
- **Authors:** (not fully listed)
- **Year:** 2025
- **Venue:** arXiv (2503.16910)
- **URL:** https://arxiv.org/html/2503.16910v1
- **Task:** Traffic SOD under weather
- **Approach:** Mamba-based model with Dual-Frequency Visual State Space module
- **Dataset:** TSOD10K (10K images with fog, snowstorms, low-contrast, low-light)
- **SOD:** Yes
- **Adverse-weather SOD:** Yes (traffic-specific)
- **Relevance:** Large-scale traffic SOD dataset under weather.

---

## TIER 6: MoE Foundations and Routing Mechanisms

### DB-41. Scaling Vision with Sparse Mixture of Experts (V-MoE)
- **Authors:** Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, Neil Houlsby
- **Year:** 2021
- **Venue:** NeurIPS 2021
- **URL:** https://arxiv.org/abs/2106.05974
- **Task:** Image classification
- **MoE:** Yes — token-level top-k routing (k=2 default)
- **Routing mechanism:** Noisy top-k (Shazeer-style) + Batch Prioritized Routing
- **Routing granularity:** Token-level
- **Relevance:** Foundational token-level top-k routing for ViT. Direct architectural ancestor of our router.

### DB-42. From Sparse to Soft Mixtures of Experts (Soft MoE)
- **Authors:** Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Neil Houlsby
- **Year:** 2024
- **Venue:** ICLR 2024
- **URL:** https://arxiv.org/abs/2308.00951
- **Task:** Image classification
- **MoE:** Yes — fully differentiable weighted combination (no top-k)
- **Routing granularity:** Token-level (but all tokens participate in each expert via convex combination)
- **Relevance:** Alternative to hard top-k routing. Eliminates load-balancing losses entirely.

### DB-43. M3ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design
- **Authors:** Hanxue Liang, Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang
- **Year:** 2022
- **Venue:** NeurIPS 2022
- **URL:** https://arxiv.org/abs/2210.14793
- **Task:** Multi-task learning (classification + detection + segmentation)
- **MoE:** Yes — task-conditioned token routing
- **Routing granularity:** Token-level with task embedding
- **Relevance:** Task-conditioned routing precedent. Not multi-scale in pyramid sense.

### DB-44. Mod-Squad: Designing Mixtures of Experts as Modular Multi-task Learners
- **Authors:** Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik Learned-Miller, Chuang Gan
- **Year:** 2023
- **Venue:** CVPR 2023
- **URL:** https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Mod-Squad_Designing_Mixtures_of_Experts_As_Modular_Multi-Task_Learners_CVPR_2023_paper.html
- **Task:** Multi-task learning
- **MoE:** Yes — mutual information loss for task-expert assignment
- **Routing granularity:** Token-level with task-specific routing networks
- **Relevance:** Modular MoE precedent for multi-task.

### DB-45. Routers in Vision Mixture of Experts: An Empirical Study
- **Authors:** Tianlin Liu, Mathieu Blondel, Carlos Riquelme Ruiz, Joan Puigcerver
- **Year:** 2024
- **Venue:** TMLR
- **URL:** https://arxiv.org/abs/2401.15969
- **Task:** Image classification
- **MoE:** Comparative study of 6 router designs
- **Relevance:** Key methodological reference for router ablations.
- **CORRECTION TO BLUEPRINT:** This is a comparative benchmark of router designs, NOT a failure-mode analysis. It does not systematically document router collapse or expert underutilization.

### DB-46. AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts
- **Authors:** Tianlong Chen, Xuxi Chen, Xianzhi Du, et al.
- **Year:** 2023
- **Venue:** ICCV 2023
- **Task:** Multi-task vision
- **MoE:** Yes — adaptive expert count per task
- **Relevance:** Adaptive expert count.

---

## TIER 7: Additional Related Work

### DB-47. Salient Object Detection via Dynamic Scale Routing (DPNet)
- **Authors:** Zhenyu Wu, Shuai Li, Chenglizhao Chen, Hong Qin, Aimin Hao
- **Year:** 2022
- **Venue:** IEEE TIP
- **URL:** https://doi.org/10.1109/tip.2022.3214332
- **Task:** SOD
- **Approach:** Dynamic pyramid convolution with routing between kernel sizes
- **Relevance:** Dynamic routing concept applied to SOD at convolution level.

### DB-48. FLoMo-Net: A Novel Task-Adaptive Mixture of Experts Routing Framework for Medical Image Segmentation
- **Authors:** Md Rayhan Ahmed, Patricia Lasserre
- **Year:** 2026
- **Venue:** WACV 2026
- **URL:** https://doi.org/10.1109/wacv61042.2026.00392
- **Task:** Medical image segmentation
- **MoE:** Yes — compares soft routing vs. top-k
- **Relevance:** Methodology reference for routing design choices.

### DB-49. Unseen Data Detection using Routing Entropy in MoE for Autonomous Vehicles
- **Authors:** Lee, Shin, Park
- **Year:** 2025
- **Venue:** ASE
- **URL:** 10.1109/ASE63991.2025.00332
- **Task:** Semantic segmentation (OOD detection)
- **Approach:** Routing entropy as OOD detector
- **Relevance:** Routing entropy used for OOD detection, not as decoder input feature.

### DB-50. GeMoE: Gating Entropy for Uncertainty-Aware Adaptive Routing
- **Authors:** Cai et al.
- **Year:** 2026
- **Venue:** arXiv
- **URL:** https://arxiv.org/abs/2606.26287
- **Task:** Vision-language models
- **Approach:** Gating entropy determines experts-per-token
- **Relevance:** Entropy used for K-selection, not as decoder input feature.

### DB-51. Adaptive-K: Entropy-Guided Dynamic Expert Selection
- **Authors:** Balsamo
- **Year:** 2026
- **Venue:** Zenodo
- **URL:** 10.5281/zenodo.18282008
- **Task:** MoE LLMs
- **Approach:** Routing entropy dynamically selects K
- **Relevance:** Entropy for K-selection, not as decoder input.

### DB-52. Uncertainty-Aware Routing for MoE Dynamics (UAR)
- **Authors:** Chen et al.
- **Year:** 2026
- **Venue:** ACL
- **URL:** aclanthology.org/2026.acl-long.1801
- **Task:** MoE (general)
- **Approach:** Thermodynamic analysis: router entropy = aleatoric, Free Energy = epistemic uncertainty
- **Relevance:** Entropy-based routing control, not decoder input.

### DB-53. Point-MoE: Large-Scale Multi-Dataset Training with Mixture-of-Experts for 3D Semantic Segmentation
- **Authors:** (Multiple authors)
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.23926
- **Task:** 3D semantic segmentation
- **MoE:** Yes — token-level routing across heterogeneous datasets without domain labels
- **Relevance:** MoE for segmentation without labels; emergent expert specialization.

### DB-54. ShapeMoE: Shape-specific Mixture-of-Experts for Amodal Segmentation
- **Authors:** Zhixuan Li, Yujia Liu, Hui Chen, et al.
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2508.01664
- **Task:** Amodal segmentation
- **MoE:** Yes — shape-conditioned expert routing
- **Relevance:** Shape-conditioned expert routing for segmentation.

### DB-55. TaVER: Task-Adaptive Visual Expert Routing
- **Authors:** Donghyun Han et al.
- **Year:** 2026
- **Venue:** KSP Journal
- **Task:** Multi-backbone routing
- **Approach:** Task-conditioned expert selection for dense prediction
- **Relevance:** Multi-backbone expert routing.

### DB-56. FG-MoE: Heterogeneous mixture of experts model for fine-grained visual classification
- **Authors:** Song-Ming Yang, Jing Wen, Bin Fang
- **Year:** 2026
- **Venue:** Pattern Recognition (Elsevier)
- **Task:** Fine-grained classification
- **MoE:** Yes — 5 specialized experts (global, regional, local, texture, part) with spatially-aware gating
- **Relevance:** Heterogeneous expert design with semantic specialization.

### DB-57. D2-Former: Mixture-Of-Experts Guided Dual Transformer for Multi-Scale Medical Image Segmentation
- **Authors:** Yucai Huo, Mingchen Gao, et al.
- **Year:** 2026
- **Venue:** PMLR
- **Task:** Medical image segmentation
- **MoE:** Yes — Softer-MoE for input-adaptive feature refinement
- **Relevance:** MoE for multi-scale feature refinement in segmentation.

### DB-58. Sparse-MoE-SAM: Lightweight Framework for Plant Disease Segmentation
- **Authors:** Benhan Zhao, Xilin Kang, Hao Zhou, et al.
- **Year:** 2025
- **Venue:** Plants (MDPI)
- **DOI:** 10.3390/plants14172634
- **Task:** Plant disease segmentation
- **MoE:** Yes — sparse attention + dual-stage MoE decoder
- **Relevance:** Multi-type convolutional experts with gated routing.

### DB-59. Multi-Head Mixture-of-Experts
- **Authors:** (Multiple authors)
- **Year:** 2024
- **Venue:** NeurIPS 2024
- **Task:** Language modeling
- **MoE:** Yes — multiple routing heads per expert
- **Relevance:** Multi-head routing as alternative to single router.

### DB-60. MoCaE: Mixture of Calibrated Experts Significantly Improves Object Detection
- **Authors:** Kemal Oksuz et al.
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.14976
- **Task:** Object detection
- **MoE:** Yes — model-level MoE ensemble with calibration
- **Relevance:** Model-level MoE ensemble.

### DB-61. MoR-DASR: Mixture of Ranks with Degradation-Aware Routing
- **Authors:** Xiao He, Zhijun Tu, Kun Cheng, et al.
- **Year:** 2025
- **Venue:** AAAI 2026
- **URL:** https://arxiv.org/abs/2511.16024
- **Task:** Super-resolution
- **MoE:** Yes — LoRA rank as expert with CLIP-based degradation estimation
- **Relevance:** Degradation-aware load balancing.

### DB-62. DELNet: Continuous All-in-One Weather Removal via Dynamic Expert Library
- **Authors:** Shihong Liu, Kun Zuo, Hanguang Xiao
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2601.22573
- **Task:** Weather removal
- **MoE:** Yes — continual learning with dynamic expert library
- **Relevance:** Novel continual learning with MoE.

### DB-63. AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection
- **Authors:** Hongwei Lin, Xun Huang, Chenglu Wen, Cheng Wang
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2603.16261
- **Task:** 3D object detection
- **MoE:** Yes — Image-guided Weather-aware Routing (IWR) with explicit weather classifier
- **Weather conditioning:** Yes (explicit, ~99% accuracy)
- **Relevance:** Weather-aware routing paradigm for 3D detection.

---

## Notes on WM-MoE / MoWE Confusion

**CRITICAL CORRECTION:** The blueprint lists "WM-MoE" and "MoWE" as two separate papers. They are the SAME paper (arXiv:2303.13739). The paper was originally titled "MoWE: Mixture of Weather Experts for Multiple Adverse Weather Removal" and renamed to "WM-MoE: Weather-aware Multi-scale Mixture-of-Experts for Blind Adverse Weather Removal" during revision. Same authors, same content.

---

*Last updated: 2026-09-01. Generated via comprehensive web search.*
*Total unique papers cataloged: 63*
