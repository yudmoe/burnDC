import torch.nn as nn
import torch
import cv2
import torchvision
from model.common import conv_bn_relu, conv_shuffle_bn_relu, convt_bn_relu
from model.stodepth_lineardecay import se_resnet34_StoDepth_lineardecay, se_resnet18_StoDepth_lineardecay,se_resnet68_StoDepth_lineardecay
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from PIL import Image
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)
# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # # # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


class get_D_diff_layer(nn.Module):
    def __init__(self,prop_kernel):
        super(get_D_diff_layer, self).__init__()

        self.kernel_size = prop_kernel
        self.number_of_neighbor = self.kernel_size*self.kernel_size
        self.num_guide = self.number_of_neighbor-1

        shift_used_convKernel_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i ==2 and j ==2:
                    continue

                temp_kernel = torch.zeros(1,self.kernel_size, self.kernel_size)
                temp_kernel[0,i,j] = -1
                temp_kernel[0,2,2] = 1
                # current_K = nn.Parameter(torch.concat(current_list,0).unsqueeze(1))
                shift_used_convKernel_list.append(temp_kernel)

        self.shift_used_convKernel_list = shift_used_convKernel_list
        tem = torch.concat(shift_used_convKernel_list,0)
        self.shift_used_convK = nn.Parameter(tem.unsqueeze(1))
        self.shift_used_convK.requires_grad = False


    def forward(self, Depth):
        # with torch.no_grad():
        shifited_Ddiff = F.conv2d(Depth, self.shift_used_convK, bias=None, stride=1, padding=int((self.kernel_size-1)/2), dilation=1)
        return shifited_Ddiff


class mySPN_affinity_inorder(nn.Module):
    def __init__(self, prop_kernel,prop_time):
        super(mySPN_affinity_inorder, self).__init__()

        
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        self.number_of_neighbor = self.prop_kernel*self.prop_kernel
        self.affinity = 'mySPN_affinity_inorder'
        self.preserve_input = True

        self.get_initialD_diff = get_D_diff_layer(prop_kernel)

        for params in self.get_initialD_diff.parameters():
            params.requires_grad = False

    def _normalize_guide(self, guide):
        abs_guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)  + 0.000001 # B 1 H W
        guide_sum = torch.sum(guide, dim=1).unsqueeze(1) # B 1 H W

        half1, half2 = torch.chunk(guide, 2, dim=1)

        half1 = half1/abs_guide_sum
        half2 = half2/abs_guide_sum
        center_guide = (1 - guide_sum/abs_guide_sum)
        normalized_guide = torch.cat([half1,center_guide,half2],dim=1)
        return normalized_guide

    def forward(self, coarse_depth, guidances, confidences, sparse_depth=None,rgb=None):
        
        # coarse_depth : [B x 1 x H x W]
        # guidances : [B x 48 x H/4 x W/4, B x 48 x H/2 x W/2, B x 48 x H/1 x W/1]
        # confidence : [B x 48 x H x W]
        # weights : [B x self.args.prop_time/3 x H/4 x W/4, B x self.args.prop_time/3 x H/2 x W/2,B x self.args.prop_time/3 x H/1 x W/1]
        
        """
        分别用 1/4分辨率affinity,1/2分辨率affinity,1/1分辨率affinity各自传播:
                1/3 prop_time       1/3 prop_time     1/3prop_time    
        次数
        """
        down_sample = nn.AvgPool2d(2,stride=2)
        up_sample = nn.Upsample(scale_factor=2, mode='nearest')


        # Propagation 

        feat_result = coarse_depth

        list_feat = []
        
        feat_result = down_sample(down_sample(feat_result))
        for i in range(3):
            current_guidances = guidances[i]
            current_weights = confidences[i]
            if i==1:
                feat_result = nn.Upsample(scale_factor=2, mode='nearest')(feat_result)
            if i==2:
                feat_result = nn.Upsample(scale_factor=2, mode='nearest')(feat_result)

            iteral_time = int(self.prop_time/3)
            for j in range(iteral_time):
                const_index = i*iteral_time + j
                bs, _, h, w = feat_result.size() 
                normalized_affinity_imgsize = self._weight_guidance_byInitialD_and_norm(feat_result, current_guidances,  
                                                                                        current_weights[:,j:j+1], 
                                                                                        self.affweight_scale_const[const_index])
                B, channel, H, W = normalized_affinity_imgsize.shape
                current_aff = normalized_affinity_imgsize.reshape(B, self.number_of_neighbor , H * W)

                depth_im2col = F.unfold(feat_result, self.prop_kernel, 1, int((self.prop_kernel-1)/2), 1)
                guide_result = torch.einsum('ijk,ijk->ik', (depth_im2col, current_aff))
                propageted_depth = guide_result.view(bs, 1, h, w)

                feat_result = propageted_depth
                list_feat.append(propageted_depth)

        return feat_result, list_feat , normalized_affinity_imgsize

    def _propagation_onece(self, current_D, guidances, sparse_depth, burun_weight=None):
        # Propagation 
        # preserve_input is True
        temopp = sparse_depth[0,0]
        if self.preserve_input:
            mask_fix = torch.sum(sparse_depth > 0.0, dim=1, keepdim=True).detach()
            temopp2 = sparse_depth[0,0]
            mask_fix = (mask_fix > 0.0).type_as(sparse_depth)
            temopp3 = mask_fix[0,0]

        Depth_before_porp = current_D.clone()

        bs, _, h, w = current_D.size() 
        normalized_affinity_imgsize = self._normalize_guide(guidances)
        B, channel, H, W = normalized_affinity_imgsize.shape
        current_aff = normalized_affinity_imgsize.reshape(B, self.number_of_neighbor , H * W)

        depth_im2col = F.unfold(current_D, self.prop_kernel, 1, int((self.prop_kernel-1)/2), 1)
        guide_result = torch.einsum('ijk,ijk->ik', (depth_im2col, current_aff))
        propageted_depth = guide_result.view(bs, 1, h, w)

        if burun_weight!=None:
            mask_temp = burun_weight[0,0]
            temp = propageted_depth.clone()
            propageted_depth = Depth_before_porp*(1-burun_weight) + propageted_depth*burun_weight
            Depth_before_porp = temp

        if self.preserve_input:
            temp1 = sparse_depth[0,0]
            temp2 = propageted_depth[0,0]
            propageted_depth = (1.0 - mask_fix) * propageted_depth \
                            + mask_fix * sparse_depth
        return propageted_depth, Depth_before_porp, normalized_affinity_imgsize
    


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)
class simpleBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(simpleBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out

def get_source_area_xywh_batch(depth_map):
    """
    计算每个样本的热源区域边界，返回一个字典。
    输入：depth_map - 形状为 B * 1 * H * W 的深度图批次。
    输出：一个字典，其中每个键对应一个样本的边界坐标 (x_min, y_min, x_max, y_max)。
    """
    batch_size = depth_map.size(0)  # B, 获取批次大小
    result = {}

    # 逐个样本处理
    for b in range(batch_size):
        binary_map = (depth_map[b, 0] > 0).float()  # 获取当前样本的二值化图像，热源区域为1
        x_indices, y_indices = torch.nonzero(binary_map, as_tuple=True)  # 找到热源区域的坐标
        # 如果没有热源区域，则返回默认的边界值
        if x_indices.numel() == 0 or y_indices.numel() == 0:
            result[b] = {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0}
        else:
            try:
                # 获取热源区域的边界
                x_min, x_max = torch.min(x_indices), torch.max(x_indices)
                y_min, y_max = torch.min(y_indices), torch.max(y_indices)

                # 将边界信息存储到字典中
                result[b] = {'x_min': x_min.item(), 'y_min': y_min.item(), 'x_max': x_max.item(), 'y_max': y_max.item()}
            except Exception as e:
                # 如果出错则设置为默认值
                result[b] = {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0}
                print(f"Error for batch {b}: {e}")

    return result

def fill_empty_regions(depth_map):
    H, W = depth_map.shape[1], depth_map.shape[2]
    device = depth_map.device
    result = depth_map.clone()

    # 横向填充优化
    mask_row = result[0] > 0
    has_nonzero_row = mask_row.any(dim=1)
    
    # 计算首尾非零索引
    left_indices = torch.argmax(mask_row.int(), dim=1)
    right_indices = (W - 1) - torch.argmax(mask_row.flip(1).int(), dim=1)
    
    # 生成列坐标网格
    col_grid = torch.arange(W, device=device).expand(H, W)
    
    # 构建掩码并填充
    left_mask = (col_grid < left_indices.unsqueeze(-1)) & has_nonzero_row.unsqueeze(-1)
    right_mask = (col_grid >= right_indices.unsqueeze(-1)) & has_nonzero_row.unsqueeze(-1)
    result[0] = torch.where(left_mask, result[0][torch.arange(H), left_indices][:, None], result[0])
    result[0] = torch.where(right_mask, result[0][torch.arange(H), right_indices][:, None], result[0])

    # 竖向填充优化
    mask_col = result[0] > 0
    has_nonzero_col = mask_col.any(dim=0)
    
    # 计算首尾非零索引
    top_indices = torch.argmax(mask_col.int(), dim=0)
    bottom_indices = (H - 1) - torch.argmax(mask_col.flip(0).int(), dim=0)
    
    # 生成行坐标网格
    row_grid = torch.arange(H, device=device).expand(W, H).T
    
    # 构建掩码并填充
    top_mask = (row_grid < top_indices[None, :]) & has_nonzero_col[None, :]
    bottom_mask = (row_grid >= bottom_indices[None, :]) & has_nonzero_col[None, :]
    result[0] = torch.where(top_mask, result[0][top_indices, torch.arange(W)], result[0])
    result[0] = torch.where(bottom_mask, result[0][bottom_indices, torch.arange(W)], result[0])

    return result

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes, reduction=16)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DMRe(nn.Module):
    def __init__(self, feature_channel, depth_channel):
        super(DMRe, self).__init__()
        # Conv-BN-ReLU with the requested 1/4 scale feature and channel concat (instead of 1/2 scale feature)
        self.dep_initial = conv_bn_relu(1, depth_channel, kernel=3, stride=1, padding=1, bn=False)
        
        # Channel Attention (feature_channel + depth_channel)
        self.channel_wise_attention = ChannelAttention(feature_channel + depth_channel, reduction=16)
        
        # Spatial Attention
        self.spatial_attention = SpatialAttention()
         # Conv to reduce channels from (feature_channel + depth_channel) to feature_channel
        self.channel_reduction = nn.Conv2d(feature_channel + depth_channel, feature_channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, feature, depth):
        # Process depth input through the initial convolution
        dep_initial = self.dep_initial(depth)
        
        # Concatenate feature and processed depth information
        combined_feature = torch.cat((feature, dep_initial), dim=1)
        
        # Apply channel attention
        combined_feature = combined_feature * self.channel_wise_attention(combined_feature)
        
        # Apply spatial attention
        combined_feature = self.spatial_attention(combined_feature) * combined_feature
        
        # Reduce channels to feature_channel
        feature = self.channel_reduction(combined_feature)

        return feature


def compute_distance_map(H, W, x_min, y_min, x_max, y_max, device='cuda'):
    """
    根据矩形 (x_min, y_min, x_max, y_max) 生成每个点的距离图
    输入：H, W: 图像的高度和宽度
        x_min, y_min, x_max, y_max: 矩形的左下角和右上角坐标
        device: 设备类型，默认为'cuda'，即使用GPU
    输出：distance_map: 大小为 H*W 的距离图
    """
    # 将输入转移到GPU
    x_coords, y_coords = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device), 
                                         torch.arange(W, dtype=torch.float32, device=device))
    
    # 初始化距离图
    distance_map = torch.zeros(H, W, dtype=torch.float32, device=device)
    
    # 1. 矩形内部的情况：计算该点坐标与矩形四个边的最小距离
    distance_to_boundary = torch.minimum(torch.abs(x_coords - x_min), torch.abs(x_coords - x_max))
    distance_to_boundary = torch.minimum(distance_to_boundary, torch.minimum(torch.abs(y_coords - y_min), torch.abs(y_coords - y_max)))
    
    distance_map[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = 0
    distance_map[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = distance_to_boundary[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)]
    
    # 2. 矩形的正上下左右区域
    distance_map[(x_coords < x_min) & (y_coords >= y_min) & (y_coords <= y_max)] = x_min - x_coords[(x_coords < x_min) & (y_coords >= y_min) & (y_coords <= y_max)]
    distance_map[(x_coords > x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = x_coords[(x_coords > x_max) & (y_coords >= y_min) & (y_coords <= y_max)] - x_max
    distance_map[(y_coords < y_min) & (x_coords >= x_min) & (x_coords <= x_max)] = y_min - y_coords[(y_coords < y_min) & (x_coords >= x_min) & (x_coords <= x_max)]
    distance_map[(y_coords > y_max) & (x_coords >= x_min) & (x_coords <= x_max)] = y_coords[(y_coords > y_max) & (x_coords >= x_min) & (x_coords <= x_max)] - y_max

    # 3. 矩形的角落区域：计算点到矩形四个角的距离
    dist_tl = torch.sqrt((x_coords - x_min)**2 + (y_coords - y_min)**2)  # 距离左上角
    dist_tr = torch.sqrt((x_coords - x_max)**2 + (y_coords - y_min)**2)  # 距离右上角
    dist_bl = torch.sqrt((x_coords - x_min)**2 + (y_coords - y_max)**2)  # 距离左下角
    dist_br = torch.sqrt((x_coords - x_max)**2 + (y_coords - y_max)**2)  # 距离右下角

    # 获取四个角的最小距离
    min_corner_dist = torch.min(dist_tl, torch.min(dist_tr, torch.min(dist_bl, dist_br)))

    # 角落区域：更新最小距离
    distance_map[(x_coords < x_min) & (y_coords < y_min)] = torch.maximum(distance_map[(x_coords < x_min) & (y_coords < y_min)], dist_tl[(x_coords < x_min) & (y_coords < y_min)])
    distance_map[(x_coords > x_max) & (y_coords < y_min)] = torch.maximum(distance_map[(x_coords > x_max) & (y_coords < y_min)], dist_tr[(x_coords > x_max) & (y_coords < y_min)])
    distance_map[(x_coords < x_min) & (y_coords > y_max)] = torch.maximum(distance_map[(x_coords < x_min) & (y_coords > y_max)], dist_bl[(x_coords < x_min) & (y_coords > y_max)])
    distance_map[(x_coords > x_max) & (y_coords > y_max)] = torch.maximum(distance_map[(x_coords > x_max) & (y_coords > y_max)], dist_br[(x_coords > x_max) & (y_coords > y_max)])

    return distance_map

def compute_direction_map(H, W, x_min, y_min, x_max, y_max, device='cuda'):
    """
    计算传播方向图，分区域进行方向计算：1. 对于左上、左下、右上、右下区域：传播方向是当前点和矩形中心的连线。2. 对于正上、正下、正左、正右区域：传播方向垂直于最近的矩形边界。
    返回一个H * W * 2的张量，表示每个点的传播方向（单位向量）。
    """
    # 将输入转移到GPU
    x_coords, y_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    
    # 计算矩形中心
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # 初始化方向图
    direction_map = torch.zeros(H, W, 2, dtype=torch.float32, device=device)
    
    # 计算点到矩形中心的相对位置
    relative_x = x_coords - center_x
    relative_y = y_coords - center_y
    
    # 左上、左下、右上、右下区域：单位向量
    direction_map[(x_coords < x_min) & (y_coords < y_min)] = torch.stack(
        [(relative_x / torch.sqrt(relative_x**2 + relative_y**2 +1e-3)), 
         (relative_y / torch.sqrt(relative_x**2 + relative_y**2 +1e-3))], dim=-1)[(x_coords < x_min) & (y_coords < y_min)]
    
    direction_map[(x_coords > x_max) & (y_coords < y_min)] = torch.stack(
        [(relative_x / torch.sqrt(relative_x**2 + relative_y**2 +1e-3)), 
         (relative_y / torch.sqrt(relative_x**2 + relative_y**2 +1e-3))], dim=-1)[(x_coords > x_max) & (y_coords < y_min)]
    
    direction_map[(x_coords < x_min) & (y_coords > y_max)] = torch.stack(
        [(relative_x / torch.sqrt(relative_x**2 + relative_y**2 +1e-3)), 
         (relative_y / torch.sqrt(relative_x**2 + relative_y**2 +1e-3))], dim=-1)[(x_coords < x_min) & (y_coords > y_max)]
    
    direction_map[(x_coords > x_max) & (y_coords > y_max)] = torch.stack(
        [(relative_x / torch.sqrt(relative_x**2 + relative_y**2 +1e-3)), 
         (relative_y / torch.sqrt(relative_x**2 + relative_y**2 +1e-3))], dim=-1)[(x_coords > x_max) & (y_coords > y_max)]

    # 正上、正下、正左、正右区域：单位向量沿着边缘法向量
    direction_map[(y_coords < y_min) & (x_coords >= x_min) & (x_coords <= x_max)] = torch.tensor([0, -1], dtype=torch.float32, device=device)
    direction_map[(y_coords > y_max) & (x_coords >= x_min) & (x_coords <= x_max)] = torch.tensor([0, 1], dtype=torch.float32, device=device)
    direction_map[(x_coords < x_min) & (y_coords >= y_min) & (y_coords <= y_max)] = torch.tensor([-1, 0], dtype=torch.float32, device=device)
    direction_map[(x_coords > x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = torch.tensor([1, 0], dtype=torch.float32, device=device)
    
    direction_map[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = torch.stack(
        [(relative_x / torch.sqrt(relative_x**2 + relative_y**2 +1e-3)), 
         (relative_y / torch.sqrt(relative_x**2 + relative_y**2 +1e-3))], dim=-1)[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)]

    return direction_map



def compute_ideal_direction_and_strength(B, H, W, source_fileds_bounds, sigma=3, device='cuda'):
    """
    中心区域对应的四个分辨率分别是:[[77,227],[39,189]], [[39,114],[20,95]], [[20,57],[10,48]], [[10,29],[5,24]]
    """
    # 初始化返回结果
    # direction_map_batch = torch.zeros((B, H, W, 2), dtype=torch.float32, device=device)  # B * H * W * 2
    strength_map_batch = torch.zeros((B, H, W), dtype=torch.float32, device=device)  # B * H * W

    current_xyxy = source_fileds_bounds
    x_min,  x_max, y_min, y_max = current_xyxy[1][0], current_xyxy[1][1], current_xyxy[0][0], current_xyxy[0][1]
    # 计算传播方向和强度
    # direction_map = compute_direction_map(H, W, x_min, y_min, x_max, y_max, device=device)
    rectangle_distance = compute_distance_map(H, W, x_min, y_min, x_max, y_max, device=device)

    # 归一化距离图到 [0, 1] 范围
    normalized_distance_map = (rectangle_distance / H) * 25
    rectangle_distance = normalized_distance_map

    # 计算传播强度
    strength_map = torch.exp(-rectangle_distance**2 / (2 * sigma**2))

    # 遍历每个样本，计算传播方向和强度
    for b in range(B):
        # 存储结果
        # direction_map_batch[b] = direction_map
        strength_map_batch[b] = strength_map

    strength_map_batch[:,  x_min:x_max , y_min:y_max ] = 0 #去除中心区域(因为中心区域直接用原始深度图)

    center_mask = torch.zeros((B, H, W), dtype=torch.float32, device=device)  # B * H * W
    center_mask[:,  x_min:x_max , y_min:y_max ] = 1 #去除中心区域(因为中心区域直接用原始深度图)

    return strength_map_batch.unsqueeze(1), center_mask.unsqueeze(1)
    return direction_map_batch, strength_map_batch



def expand_area_by_ratio(x_min, y_min, x_max, y_max, expand_ratio_x, expand_ratio_y, depth_map_size):
    """
    根据给定的扩展比例和膨胀次数，膨胀长方形区域的边界。
    输入：
        x_min, y_min, x_max, y_max - 当前长方形区域的边界
        expand_ratio_x, expand_ratio_y - 在x和y方向上的膨胀比例
        depth_map_size - 深度图的尺寸，用于确保边界不会超出图像范围 (H, W)
        times - 膨胀次数
    输出：
        扩展后的边界
    """
    H, W = depth_map_size

    # 计算当前区域的宽度和高度
    current_width = x_max - x_min
    current_height = y_max - y_min

    # 计算膨胀的增量
    expand_value_x = current_width * (expand_ratio_x - 1)
    expand_value_y = current_height * (expand_ratio_y - 1)

    # 扩展区域边界
    x_min = max(0, x_min - expand_value_x)
    y_min = max(0, y_min - expand_value_y)
    x_max = min(H - 1, x_max + expand_value_x)
    y_max = min(W - 1, y_max + expand_value_y)

    return int(x_min), int(y_min), int(x_max), int(y_max)

def get_ideal_confi(center_square_depth, center_mask=[[77,227],[39,189]], Sigma=3):
    """"
    中心区域对应的四个分辨率分别是:[[77,227],[39,189]], [[39,114],[20,95]], [[20,57],[10,48]], [[10,29],[5,24]]
    """

    H, W = center_square_depth.shape[2], center_square_depth.shape[3]
    # expand_ratio_x = 1 + 0.05 * expand_stage
    # expand_ratio_y = 1 + 0.08 * expand_stage
    # x_min_exp, y_min_exp, x_max_exp, y_max_exp = expand_area_by_ratio(
    #                 x_min.item(), y_min.item(), x_max.item(), y_max.item(), expand_ratio_x, expand_ratio_y, (H, W))
    # expanded_xyxy = {'x_min': x_min_exp, 'y_min': y_min_exp, 'x_max': x_max_exp, 'y_max': y_max_exp}

    batch_xywh = center_mask
    ideal_strength,  center_mask= compute_ideal_direction_and_strength(center_square_depth.size(0), 
                                                                           center_square_depth.size(2), 
                                                                           center_square_depth.size(3), 
                                                                           batch_xywh,
                                                                           sigma=Sigma,
                                                                           device = center_square_depth.device)
    
    ideal_strength = ideal_strength.to(center_square_depth.device)    
    center_mask = center_mask.to(center_square_depth.device)    

    return  ideal_strength, center_mask


class HCSPN_Model(nn.Module):
    def __init__(self, prop_kernel, prop_time,data_name='NYU',norm_depth=[0.1, 10.0],  sto=True, res="res34", suffle_up=False, norm_layer=None):
        super(HCSPN_Model, self).__init__()
        self.data_name = data_name
        self.norm_depth = norm_depth
        self.prop_kernel = prop_kernel
        self.prop_time = prop_time
        self.num_guide = prop_kernel*prop_kernel -1 
        self.num_time_of_one_step = int(self.prop_time//4)
        self.half_k = int(self.prop_kernel//2)
         # Encoder
        self.conv1_rgb = conv_bn_relu(3, 36, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 14, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv_dep_initial = conv_bn_relu(1, 14, kernel=3, stride=1, padding=1,
                                      bn=False)


        if sto == True:
            if res == "res18":
                net = se_resnet18_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
            else:
                net = se_resnet34_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
                # net = se_resnet68_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
        else:
            if res == "res18":
                net = torchvision.models.resnet18(pretrained=True)
            else:
                net = torchvision.models.resnet34(pretrained=True)
        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4
        del net
        # # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)


        if suffle_up == True:
            # 1/8
            self.dec5 = conv_shuffle_bn_relu(512, 256, kernel=3, stride=1, padding=1)
            # 1/4
            self.dec4 = conv_shuffle_bn_relu(256 + 512, 128, kernel=3, stride=1, padding=1)
            # 1/2
            self.dec3 = conv_shuffle_bn_relu(128 + 256, 64, kernel=3, stride=1, padding=1)
            # 1/
            self.dec2 = conv_shuffle_bn_relu(64 + 128, 64, kernel=3, stride=1, padding=1)
        else:
            # Shared Decoder
            # # 1/8
            # self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1)
            self.dec5 =nn.Sequential(convt_bn_relu(512, 256, kernel=3, stride=2,
                                                    padding=1, output_padding=1),
                                        BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
                                    )

            # # Guidance Branch
            # # 1/8
            self.guide_quater = nn.Sequential(conv_bn_relu(512+256, 128, kernel=3, stride=1,padding=1),
                                              conv_bn_relu(128, self.num_guide, kernel=3, stride=1, padding=1, bn=False, relu=False)
                                              )

            #还是1/8尺度，但是是用来并入stage1的迭代结果和
            self.DMRE_quater = DMRe(feature_channel=256+512 , depth_channel=128)

            # 1/4
            self.dec4 = nn.Sequential(convt_bn_relu(256 + 512, 128, kernel=3, stride=2, padding=1, output_padding=1),
                                        BasicBlock(128, 128, stride=1, downsample=None, ratio=16),
                                    )

            # # Guidance Branch
            # # 1/4
            self.guide_fouth = nn.Sequential(conv_bn_relu(128+256, 128, kernel=3, stride=1,padding=1),
                                              conv_bn_relu(128, self.num_guide, kernel=3, stride=1, padding=1, bn=False, relu=False)
                                              )

            # #还是1/4尺度，但是是用来并入stage1的迭代结果和
            self.DMRE_fouth = DMRe(feature_channel=256+128 , depth_channel=128)


            # 1/2
            self.dec3 = nn.Sequential(convt_bn_relu(256+128, 128, kernel=3, stride=2,
                                    padding=1, output_padding=1),
                                        BasicBlock(128, 128, stride=1, downsample=None, ratio=16),
                                    )
            self.guide_half = nn.Sequential(conv_bn_relu(128+128, 128, kernel=3, stride=1,padding=1),
                                             conv_bn_relu(128, self.num_guide, kernel=3, stride=1,
                                        padding=1, bn=False, relu=False))
            #这个地方stage2的结果已经出来了，但是用要重新输入网络中与当前尺度的1/2特征图concat之后再过反卷积
            self.DMRE_half = DMRe(feature_channel=128+128 , depth_channel=64)

            # 1/1
            self.dec2 = nn.Sequential(convt_bn_relu(128+128, 128, kernel=3, stride=2,
                                    padding=1, output_padding=1),
                                        BasicBlock(128, 128, stride=1, downsample=None, ratio=16),
                                    )
            self.guide_full = nn.Sequential(conv_bn_relu(128+64, 128, kernel=3, stride=1,padding=1),
                                             conv_bn_relu(64+64, self.num_guide, kernel=3, stride=1,
                                        padding=1, bn=False, relu=False))
       
            self.prop_layer = mySPN_affinity_inorder(prop_kernel = self.prop_kernel, prop_time = self.prop_time)


    def _make_layer(self, block, out_channels, num_blocks):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = []
        for stride in range(num_blocks):
            layers.append(block( out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def _make_layer(self, block, out_channels, num_blocks):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = []
        for stride in range(num_blocks):
            layers.append(block( out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    def down_sample(self, depth,stride):
        C = (depth > 0).float()
        output = F.avg_pool2d(depth, stride, stride) / (F.avg_pool2d(C, stride, stride) + 0.0001)
        return  output
    
    def forward(self, rgb, dep, intial_depths):
        
        #并入mask0
        C = (dep > 0).float()
        dep_eight = F.avg_pool2d(dep, kernel_size = 8, stride = 8, padding=(2,0)) / (F.avg_pool2d(C, kernel_size = 8, stride = 8, padding=(2,0)) + 0.0001)
        burun_weight0, center_mask0 = get_ideal_confi(dep_eight, center_mask=[[10,29],[5,24]], Sigma=3.5 )
        dep_quater = self.down_sample(dep, 4)
        burun_weight1, center_mask1 = get_ideal_confi(dep_quater, center_mask=[[20,57],[10,48]], Sigma=5 )
        dep_half = self.down_sample(dep, 2)
        burun_weight2, center_mask2 = get_ideal_confi(dep_half, center_mask=[[39,114],[20,95]], Sigma=7)



        bz = dep.shape[0] 

        dep = dep 

        y_inter=[]
        y_before_inter=[]

        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe_dep_initial = self.conv_dep_initial(dep)
        fe_initial = torch.cat((fe1_rgb, fe1_dep, fe_dep_initial), dim=1)#36+14


        fe1 = fe_initial
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)#512 256 upsample 1/8
        eighth_feature_map = self._concat(fd5, fe5)
        guide3 = self.guide_quater(eighth_feature_map)


        current_depth = dep_eight
        for index in range(self.num_time_of_one_step):
            constant_index = self.num_time_of_one_step *0 + index
            current_depth, Depth_before_porp, normalized_affinity_imgsize3 =  self.prop_layer._propagation_onece(current_depth, guide3, dep_eight, burun_weight=burun_weight0)
            y_inter.append(current_depth)
            y_before_inter.append(Depth_before_porp)
        y3 = current_depth

        #stage1的后处理
        feature_initial2 = self.DMRE_quater(eighth_feature_map, y3)

        fd4 = self.dec4(feature_initial2)#256+512 128 upsample 1/4

        # Guidance Decoding 1/4
        quater_feature_map = self._concat(fd4, fe4)

        guide2 = self.guide_fouth(quater_feature_map)

        dep_quater = self.down_sample(dep, 4)
        intial_depths2 = nn.AdaptiveAvgPool2d((guide2.size(2),guide2.size(3)))(y3)
        current_depth = intial_depths2
        for index in range(self.num_time_of_one_step):
            constant_index = self.num_time_of_one_step *1 + index
            current_depth, Depth_before_porp, normalized_affinity_imgsize2 =  self.prop_layer._propagation_onece(current_depth, guide2, dep_quater, burun_weight=burun_weight1)
            y_inter.append(current_depth)
            y_before_inter.append(Depth_before_porp)
        y2 = current_depth

        #stage1的后处理
        feature_initial1 = self.DMRE_fouth(quater_feature_map, y2)

        #1/2
        fd3 = self.dec3(feature_initial1)#128+256+128 64 upsample 1/2 从1/4尺度到1/2尺度
        half_feature_map =  self._concat(fd3, fe3)

        guide1 = self.guide_half(half_feature_map)

        dep_half = self.down_sample(dep, 2)
        intial_depths1 = nn.Upsample(scale_factor=2, mode='bilinear')(y2)
        current_depth = intial_depths1
        for index in range(self.num_time_of_one_step):
            constant_index = self.num_time_of_one_step *2 + index
            current_depth, Depth_before_porp, normalized_affinity_imgsize1 =  self.prop_layer._propagation_onece(current_depth, guide1, dep_half, burun_weight=burun_weight2)
            y_inter.append(current_depth)
            y_before_inter.append(Depth_before_porp)

        y1 = current_depth

        feature_initial0 = self.DMRE_half(half_feature_map, y1)
        
        #1/1
        fd2 = self.dec2(feature_initial0)#64+128 64 upsample 1/1
        full_feature = torch.cat((fd2, fe2), dim=1)
        guide0 = self.guide_full(full_feature)

        dep_full = dep
        intial_depths0 = nn.Upsample(scale_factor=2, mode='bilinear')(y1)
        current_depth = intial_depths0
        for index in range(self.num_time_of_one_step):
            constant_index = self.num_time_of_one_step *3 + index
            current_depth, Depth_before_porp, normalized_affinity_imgsize0 =  self.prop_layer._propagation_onece(current_depth, guide0, dep_full)
            y_inter.append(current_depth)
            y_before_inter.append(Depth_before_porp)


        y = current_depth

        

        y_inter = [inter  for inter in y_inter]
        y = y  
        intial_depths = dep_eight  
        # Remove negative depth
        y = torch.clamp(y, min=self.norm_depth[0],max=self.norm_depth[1])

        return {'pred': y,
                'pred_init': intial_depths,
                "list_feat": y_inter,
                "y_before_inter": y_before_inter,
                "offset": normalized_affinity_imgsize0,
                "aff": normalized_affinity_imgsize0
                }


from thop import profile
# 增加可读性
from thop import clever_format
import time
from icecream import ic
class Model(nn.Module):
    def __init__(self, data_name='NYU',iteration=3, num_neighbor=9, mode="dyspn", shuffle_up=False, norm_depth=[0.1, 10.0], res="res18",
                 bm="v1", norm_layer='bn', stodepth=True):
        super(Model, self).__init__()
        self.sto = stodepth
        # self.sto = False
        assert res in ["res18", "res34"]
        self.res = res
        self.bm = bm
        self.shuffle_up = shuffle_up
        self.mode = mode
        assert norm_layer in ['bn', 'in']
        self.norm_layer = norm_layer
        self.iteration = iteration
        self.num_sample = num_neighbor
        BM = HCSPN_Model
        self.base = BM(data_name=data_name, norm_depth=norm_depth, prop_time=self.iteration, prop_kernel=self.num_sample)
        
        self.norm = norm_depth

        self.cost_time_list = []

    def forward(self, rgb0, dep, prefilled):


        # torch.cuda.synchronize()
        # t0 = time.time()

        otuput = self.base(rgb = rgb0, dep = dep, intial_depths = prefilled)

        # torch.cuda.synchronize()
        # self.cost_time_list.append(time.time()-t0)
        # avg_cost = sum(self.cost_time_list)/len(self.cost_time_list)
        # if len(self.cost_time_list)%100==0:
        #     ic(avg_cost)

        #     flops, params = profile(self.base, inputs=(rgb0, dep, prefilled,))
        #     flops, params = clever_format([flops, params], "%.3f")  
        #     ic(flops,   params )
        return otuput
