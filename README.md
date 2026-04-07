# BurnDC

**Official PyTorch implementation of "BurnDC: A Progressive Propagation Framework for Low Coverage Depth Completion"**

---

## 🎥 Supplementary Visualizations

We provide supplementary animated visualizations to further demonstrate the effectiveness and robustness of BurnDC under the LCDC setting. To avoid concerns about cherry-picking, the examples shown here are not manually selected, but are sampled from the evaluation sets according to fixed rules.


### 1. LC-KITTI Multi-Scene Visualization

![LC-KITTI-multiscene](lc_kitti_scene_samples.gif)

*This visualization presents multiple continuous scenes from LC-KITTI to illustrate the temporal consistency and robustness of BurnDC in realistic driving scenarios. 
The displayed sequences follow a fixed sampling protocol instead of manual selection, helping provide a more representative qualitative evaluation.*


### 2. LC-NYU Progressive Completion

![LC-NYU-progressive](nyu_progressive_burn.gif)

*This visualization presents the progressive depth completion process on LC-NYU. The displayed samples are uniformly sampled from the test set at a fixed interval, rather than manually chosen. 
For compact visualization, two samples are shown side by side in each sequence, with their original indices marked in the upper-left corner. The intermediate predictions are rendered as point clouds to highlight the gradual recovery of scene geometry.*

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


