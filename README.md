# BurnDC

**Official PyTorch implementation of "BurnDC: A Progressive Propagation Framework for Low Coverage Depth Completion"**

---

## 🎥 Supplementary Visualizations

We provide animated visualizations to demonstrate the effectiveness and robustness of BurnDC in recovering dense geometry from low-coverage inputs.

### 1. NYU Progressive Recovery

<video src="NYU-progressive-burn.mp4" autoplay loop muted playsinline width="600"></video>

*This visualization illustrates the **progressive 24-step depth recovery process** on the LC-NYU dataset. The intermediate depth predictions are rendered as point clouds to highlight the stable geometric expansion.*

---


### 2. KITTI Dynamic Sequence
![KITTI-dynamic](KITTI-dynamic.gif)
*A **dynamic 50-frame continuous sequence** from the LC-KITTI dataset, demonstrating BurnDC's temporal consistency and robustness in real-world driving scenarios.*

---

## 🛠️ How to Run

Follow these steps to set up and run the framework:

1. **Dataset Configuration:** Modify the dataset path in `datasetsettings_NYU.py` to match your local environment.
2. **Training Setup:** Open `settings_NYU.py` and set `test_only = False` to enable the training mode.
3. **Execution:** Launch the main script using:
   ```bash
   python burn_DC_main.py

## 🚀 Future Plans & Updates

We are committed to maintaining and improving this repository. Upcoming updates include:
- [ ] **Pre-trained Weights:** Release of model weights for LC-NYU, LC-KITTI, and LC-TIERS.
- [ ] **Dataset Support:** Comprehensive instructions for reproducing TIERS benchmarks.
- [ ] **Advanced Boundary Models:** Exploration of non-rectangular propagation envelopes.

*We will continue to update this codebase as the research progresses. Stay tuned!*

---


