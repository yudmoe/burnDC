# BurnDC

Official implementation of BurnDC

---

## Supplementary Visualizations

We provide two animated GIFs to further demonstrate the effectiveness and robustness of BurnDC:

### 1. NYU Progressive Recovery

![NYU-progressive-burn](NYU-progressive-burn.gif)

Progressive 24-step depth recovery process on the LC-NYU dataset with point cloud visualization.

### 2. KITTI Dynamic Sequence

![KITTI-dynamic](KITTI-dynamic.gif)

Dynamic 50-frame continuous sequences on the KITTI dataset for real driving scene evaluation.

These animations can be directly viewed and played online on this GitHub page.

---

## How to Run

1. Modify the dataset path in `datasetsettings_NYU.py` according to your local environment.

2. Set `test_only = False` in `settings_NYU.py` to enable training.

3. Run the main script: `python main.py`
