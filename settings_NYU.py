model_name = "baseline_burnmask"


seed = 3407
n_device = 2
# dataset
dataset_py =  'nyu_incomplete_centersquare_sampling'
data_name = "NYU"
dir_data = '/data/zzy/NYU_DepthV2_HDF5/nyudepthv2'
split_json = "/home/zzy/ral_burnDC/nyu.json"
augment = True
num_sample = 500
patch_height = 228
patch_width = 304
# base network
norm_depth = [0.2, 10.0]
basemodel = "v1"
resnet = "res34"
sto_depth = True
pretrain_weight = None
val_output = True
resume_weight = None
# SPN
spn_enable = True
spn_module = 'model' 
prop_kernel = 5
prop_time = 24
# loss
loss_name = "sloss_4stage_Ploss_4stage"
downLR1 = 40
downLR2 = 60
w_1 = 1.0
w_2 = 1.0
# dataloader
n_thread = 8
n_batch = 24
# met
eval_range = None
# optimizer
learning_rates = 5e-4
w_weight_decay = 0
epochs = 100
step = 40
LR_down_gamma = 0.5
#test
test_only = False
if test_only:
    n_device = 1
    top_crop = 0
    pretrain_weight = None
    resume_weight = None