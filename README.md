

![CASTLE 標誌](assets/logo.png)
[![PyPI version](https://badge.fury.io/py/castle-ai.svg)](https://badge.fury.io/py/castle-ai)
[![PyPI Downloads](https://static.pepy.tech/badge/castle-ai)](https://pepy.tech/projects/castle-ai)
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CASTLE-ai/castle-ai/blob/main/notebooks/colab.ipynb)

![CASTLE 流程圖](assets/Flowchart.png)


CASTLE (Combined Approach for Segmentation and Tracking with Latent Extraction) is a training-free framework that combines segmentation models, tracking algorithms, and visual foundation models to automatically discover animal behaviors from video. Through focused latent extraction and hierarchical clustering, it achieves expert-level accuracy across multiple species without manual labeling, while uncovering previously hidden behavioral patterns that keypoint methods miss.

<p align="center">
  <img src="assets/Reaching_demo.gif" alt="Reaching Demo">
</p>

## Licenses and Acknowledgments
Licenses for borrowed code can be found in the [LICENSES](LICENSE.txt) file.

- SAM (Segment Anything Model) - https://github.com/facebookresearch/segment-anything
- DeAOT (Decoupling Features in Hierarchical Propagation) - https://github.com/yoxu515/aot-benchmark
- DINOv2 (Self-Supervised Vision Transformer) - https://github.com/facebookresearch/dinov2