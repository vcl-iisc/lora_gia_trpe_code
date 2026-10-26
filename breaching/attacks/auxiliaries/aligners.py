import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.models as models
import torch.nn as nn

class TinyFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 → 16

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16 → 8

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def build_feature_net(device):
    net = TinyFeatureNet().to(device)
    net.eval()
    return net


def dense_feature_matches(fs, ft):
    """
    fs, ft: (C, Hf, Wf) torch tensors
    returns:
        pts_s, pts_t: (N, 2) numpy arrays
    """
    C, Hf, Wf = fs.shape

    fs_flat = fs.view(C, -1).T  # (N, C)
    ft_flat = ft.view(C, -1).T

    fs_flat = F.normalize(fs_flat, dim=1)
    ft_flat = F.normalize(ft_flat, dim=1)

    sim = fs_flat @ ft_flat.T   # (N, N)
    idx = sim.argmax(dim=1)

    ys, xs = torch.meshgrid(
        torch.arange(Hf),
        torch.arange(Wf),
        indexing="ij"
    )

    pts_s = torch.stack([xs.flatten(), ys.flatten()], dim=1)
    pts_t = torch.stack([
        xs.flatten()[idx],
        ys.flatten()[idx]
    ], dim=1)

    return pts_s.cpu().numpy(), pts_t.cpu().numpy()

def estimate_homography(pts_s, pts_t, Hf, Wf, H, W):
    # scale feature coords → image coords
    scale_x = W / Wf
    scale_y = H / Hf

    pts_s = pts_s.astype(np.float32)
    pts_t = pts_t.astype(np.float32)

    pts_s[:, 0] *= scale_x
    pts_s[:, 1] *= scale_y
    pts_t[:, 0] *= scale_x
    pts_t[:, 1] *= scale_y

    if pts_s.shape[0] < 3:
        return None

    H_mat, _ = cv2.findHomography(
        pts_s,
        pts_t,
        cv2.RANSAC,
        ransacReprojThreshold=4.0
    )

    return H_mat

def warp_homography(x, H_mat):
    """
    x: (1, C, H, W) torch tensor
    H_mat: (3, 3) numpy
    """
    x_np = (
        x.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    H, W, _ = x_np.shape
    warped = cv2.warpPerspective(
        x_np,
        H_mat,
        (W, H),
        flags=cv2.INTER_LINEAR
    )

    warped = torch.from_numpy(warped).permute(2, 0, 1).unsqueeze(0)
    return warped.to(x.device)


class RANSACHomographyAligner:
    def __init__(self, netFeatCoarse, device):
        self.netFeat = netFeatCoarse
        self.device = device

    @torch.no_grad()
    def __call__(self, x_src, x_tgt):
        B, C, H, W = x_src.shape
        aligned = []

        for b in range(B):
            Is = x_src[b:b+1].to(self.device)
            It = x_tgt[b:b+1].to(self.device)

            fs = self.netFeat(Is)[0]
            ft = self.netFeat(It)[0]

            _, Hf, Wf = fs.shape

            pts_s, pts_t = dense_feature_matches(fs, ft)
            H_mat = estimate_homography(pts_s, pts_t, Hf, Wf, H, W)

            if H_mat is None:
                aligned.append(Is.cpu())
            else:
                aligned.append(warp_homography(Is, H_mat).cpu())

        return torch.cat(aligned, dim=0)


