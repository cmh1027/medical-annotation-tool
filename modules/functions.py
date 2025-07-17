import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
import pydicom
import os
from glob import glob
from scipy.ndimage import zoom
import nibabel as nib
def keep_largest_component(mask, mode, num_components=1, direction=None):
    assert len(mask.shape) == 3
    if mode == "None":
        return mask
    elif mode == "3D":
        labeled, _ = label(mask)
        props = regionprops(labeled)
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        output = np.zeros_like(mask)
        for i in range(min(num_components, len(props_sorted))):
            output[labeled == props_sorted[i].label] = 1
        return output.astype(mask.dtype)
    elif mode == "2D":
        assert direction is not None
        output = np.zeros_like(mask)
        positive_slices = np.unique(np.argwhere(mask)[..., 0])
        if direction == 'all':
            z_mid = mask.shape[0] // 2
            mask_left = mask.copy()
            mask_right = mask.copy()
            mask_left[z_mid:] = 0
            mask_right[:z_mid] = 0
            output_left = keep_largest_component(mask_left, mode, direction='left')
            output_right = keep_largest_component(mask_right, mode, direction='right')
            output = np.logical_or(output_left, output_right)
        else:
            if direction == 'left':
                positive_slices = positive_slices[::-1]
            last_slice = np.ones_like(mask[0])
            for s in positive_slices:
                if mask[s].sum() == 0:
                    break
                if (last_slice * mask[s]).sum() == 0:
                    break
                labeled, _ = label(mask[s])
                for k in np.unique(labeled):
                    if k == 0: continue
                    if ((labeled == k) * last_slice).sum() == 0:
                        labeled[labeled == k] = 0
                labeled, _ = label(labeled > 0)
                props = regionprops(labeled)
                props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
                output_slice = np.zeros_like(mask[s])
                for i in range(min(num_components, len(props_sorted))):
                    output_slice[labeled == props_sorted[i].label] = 1
                output[s] = output_slice
                last_slice = output_slice
        return output.astype(mask.dtype)
    else:
        print("Keep_largest_component : Invalid mode", mode)
        return mask

def read_dicom(dcm_path, rescale=False):
    dcm = pydicom.dcmread(dcm_path,force=True)
    if not hasattr(dcm.file_meta,'TransferSyntaxUID'):
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # type: ignore
    required_elements = ['PixelData', 'BitsAllocated', 'Rows', 'Columns',
                     'PixelRepresentation', 'SamplesPerPixel','PhotometricInterpretation',
                        'BitsStored','HighBit']
    missing = [elem for elem in required_elements if elem not in dcm]
    for elem in required_elements:
        if elem not in dcm:
            if elem== 'BitsAllocated':
                dcm.BitsAllocated = 16
            if elem== 'PixelRepresentation':
                dcm.PixelRepresentation = 1
            if elem== 'SamplesPerPixel':
                dcm.SamplesPerPixel = 1
            if elem== 'PhotometricInterpretation':
                dcm.PhotometricInterpretation = 'MONOCHROME2'
            if elem== 'BitsStored':
                dcm.BitsStored = 12
            if elem== 'BitsStored':
                dcm.BitsStored = 11
    if rescale: # For CT
        arr = dcm.pixel_array
        arr_hu = (arr * dcm.RescaleSlope + dcm.RescaleIntercept).astype(np.int16)
        dcm.PixelRepresentation = 1
        dcm.PixelData = arr_hu.tobytes()
    else:
        if dcm.PixelRepresentation == 1:
            arr = dcm.pixel_array
            overflow_threshold = 1 << (dcm.BitsStored-1)
            arr[arr >= overflow_threshold] = 0
            dcm.PixelData = arr.tobytes()
    return dcm

def read_dicoms(dcm_path, 
                slice_first=False, 
                autoflip=False, 
                return_metadata=False,
                metadata_list=[],
                rescale=False):
    dcm_list = sorted(glob(os.path.join(dcm_path, "*.dcm")))
    dcm_array = []
    instance_num = []
    sl_loc = []
    dcm_infos = []
    for sl in range(len(dcm_list)):
        dcm_info = read_dicom(dcm_list[sl], rescale=rescale)
        dcm_infos.append(dcm_info)
        dcm_array.append(dcm_info.pixel_array)
        instance_num.append(dcm_info.InstanceNumber)
        sl_loc.append(np.float16(dcm_info.ImagePositionPatient[2]))

    dcm_array  = np.array(dcm_array)
    dcm_array[np.isnan(dcm_array)] = 0
    sort_idx = np.argsort(instance_num)
    dcm_array = np.array(dcm_array)[sort_idx]
    sl_loc = np.array(sl_loc)[sort_idx]
    dcm_infos = np.array(dcm_infos)[sort_idx]

    flip_idx = sl_loc[0] > sl_loc[-1]
    if flip_idx and autoflip:
        dcm_array = np.flip(dcm_array,0)
    
    if not slice_first:
        dcm_array = np.transpose(dcm_array,(1,2,0))

    pixel_spacing = dcm_infos[0].PixelSpacing
    if hasattr(dcm_infos[0],'SpacingBetweenSlices'):
        slice_spacing = dcm_infos[0].SpacingBetweenSlices
    else:
        slice_spacing = abs(dcm_infos[1].ImagePositionPatient[2] - dcm_infos[0].ImagePositionPatient[2])
    if hasattr(dcm_infos[0], 'SliceThickness'):
        thickness = dcm_infos[0].SliceThickness
    else:
        thickness  = dcm_infos[0].SpacingBetweenSlices

    return_array = dcm_array

    if return_metadata:
        metadata = {
            "pixel_spacing" : (float(pixel_spacing[0]), float(pixel_spacing[1])),
            "slice_spacing" : float(slice_spacing),
            "thickness" : float(thickness),
            "flip_idx": flip_idx
        }
        for tag in metadata_list:
            L = []
            for dcm_info in dcm_infos:
                L.append(getattr(dcm_info, tag, None))
            metadata[tag] = L
        return return_array, metadata
    else:
        return return_array 


def apply_windowing(image_2d, center, width):
    img = image_2d.astype(np.float32)
    min_val = center - (width / 2)
    max_val = center + (width / 2)
    windowed = np.clip((img - min_val) / (max_val - min_val) * 255.0, 0, 255)
    return windowed.astype(np.uint8)

def resize(image, target_shape, order=3):
    input_shape = image.shape
    zoom_factors = [t / i for t, i in zip(target_shape, input_shape)]
    image_resized = zoom(image, zoom=zoom_factors, order=order)
    return image_resized


def modify_nifti(input_path, modification_fn=None, new_data=None, output_path=None):
    assert not (modification_fn is None and new_data is None)
    if type(input_path) is nib.nifti1.Nifti1Image:
        nifti_img = input_path
    else:
        nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    if modification_fn is None:
        modified_data = new_data
    else:
        modified_data = modification_fn(data)
    new_nifti_img = nib.Nifti1Image(modified_data.astype(np.int16), affine=nifti_img.affine, header=nifti_img.header)
    new_nifti_img.set_data_dtype(np.int16)
    if output_path is not None:
        nib.save(new_nifti_img, output_path)
    return new_nifti_img
