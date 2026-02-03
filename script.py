# cmb = (self.annotation_map == 3)
# self.annotation_map = np.where(cmb, 0, self.annotation_map)
# self.annotation_map = np.where(self.annotation_map == 2, 1, self.annotation_map)
# self.update_slice(self.slice_index)
# self.annotation_map = np.flip(self.annotation_map, 0)
# self.update_slice(self.slice_index)
# self.annotation_map = np.flip(self.annotation_map, 0)
# self.update_slice(self.slice_index)
# SB-19266-1

# cmb = (self.annotation_map == 3)
# self.annotation_map[cmb] = 0
# self.annotation_map[np.flip(cmb, 0)] = 3
# self.update_slice(self.slice_index)
from skimage.filters import frangi
from skimage.measure import label as sk_label
from skimage.measure import regionprops
import os
import numpy as np
sl = self.slice_index
if i == 1:
    self.mask = np.load(os.path.join(self.dicom_folder, "roi.npy"))
    self.logging("Load complete")
if i == 2:
    threshold = 85
elif i == 3:
    threshold = 92.5
elif i == 4:
    threshold = 95
if i == 2 or i == 3 or i == 4:
    img = self.volume[sl]
    vesselness = frangi(
        img,
        sigmas=[0.5, 0.8, 1.2, 1.6],  # SMALL scales
        alpha=0.4,   # line vs plate
        beta=0.9,    # blob suppression
        gamma=15,
        black_ridges=False  # bright ePVS on T2
    )
    candidates = vesselness > np.percentile(vesselness, threshold)
    labeled = sk_label(candidates)
    for k in np.unique(labeled):
        if k == 0: continue
        cluster = labeled == k
        if (cluster * self.mask[..., sl]).sum() < cluster.sum() / 2:
            candidates[cluster] = 0
        if max(regionprops(cluster.astype(np.uint8)), key=lambda r: r.area).major_axis_length < 5:
            candidates[cluster] = 0
    self.annotation_map[sl] = candidates
    self.update_slice(self.slice_index)