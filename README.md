# BurnDC

**Official PyTorch implementation of "BurnDC: A Progressive Propagation Framework for Low Coverage Depth Completion"**

---

## 🎥 Supplementary Visualizations

We provide supplementary animated visualizations to further demonstrate the effectiveness and robustness of BurnDC under the LCDC setting. To avoid concerns about cherry-picking, the shown examples are sampled from the evaluation sets using fixed rules rather than manual selection.


### 1. lc_kitti_scene_samples.mp4 

![LC-KITTI-multiscene](lc_kitti_scene_samples.gif)

*This visualization presents the contiguous scenes in the KITTI validation set in their original temporal order. To keep the video compact, only the first 30 frames of each scene are shown. *


### 2. nyu_progressive_burn.mp4

![LC-NYU-progressive](nyu_progressive_burn.gif)

*This visualization shows the progressive depth completion process on LC-NYU. The samples are uniformly selected from the test set at a fixed interval（50）, with two samples shown side by side and their original indices marked in the upper-left corner.*


---

## 🛠️ How to Run

Follow these steps to set up and run the framework:

1. **Dataset Configuration:** Modify the dataset path in `datasetsettings_NYU.py` to match your local environment.
2. **Training Setup:** Open `settings_NYU.py` and set `test_only = False` to enable the training mode.
3. **Execution:** Launch the main script using:
   ```bash
   python burn_DC_main.py

## 🚀 Future Plans & Updates

Upcoming updates include:
- [ ] **Pre-trained Weights:** Release of model weights for LC-NYU, LC-KITTI, and LC-TIERS.
- [ ] **Dataset Support:** Comprehensive instructions for reproducing TIERS benchmarks.
- [ ] **Advanced Boundary Models:** Exploration of non-rectangular propagation envelopes.

*We will continue to update this codebase as the research progresses. Stay tuned!*

---


