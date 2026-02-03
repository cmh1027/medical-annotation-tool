import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
import pydicom
import os
from scipy.ndimage import zoom
import nibabel as nib
import warnings
from pathlib import Path
from os.path import basename as bn
from os.path import dirname as dn
import shutil
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
                        'BitsStored','HighBit', 'RescaleSlope', 'RescaleIntercept']
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
            if elem== 'RescaleSlope':
                dcm.RescaleSlope = 1
            if elem== 'RescaleIntercept':
                dcm.RescaleIntercept = 0
    if rescale: # For CT
        arr = dcm.pixel_array
        arr_hu = (arr * dcm.RescaleSlope + dcm.RescaleIntercept).astype(np.int16)
        dcm.PixelRepresentation = 1
        dcm.PixelData = arr_hu.tobytes()
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    else:
        if dcm.PixelRepresentation == 1:
            arr = dcm.pixel_array
            overflow_threshold = 1 << (dcm.BitsStored-1)
            arr[arr >= overflow_threshold] = 0
            dcm.PixelData = arr.tobytes()
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    return dcm

def read_dicoms(dcm_path, 
                slice_first=False, 
                autoflip=False, 
                return_metadata=False,
                metadata_list=[],
                rescale=False):
    dcm_list = sorted(list(filter(lambda k:".dcm" in k.lower(), os.listdir(dcm_path))))
    dcm_array = []
    instance_num = []
    sl_loc = []
    dcm_infos = []
    instance_uids = []
    filenames = []
    for sl in range(len(dcm_list)):
        dcm_info = read_dicom(os.path.join(dcm_path,dcm_list[sl]), rescale=rescale)
        dcm_infos.append(dcm_info)
        dcm_array.append(dcm_info.pixel_array)
        instance_num.append(dcm_info.InstanceNumber)
        instance_uids.append(dcm_info.SeriesInstanceUID)
        sl_loc.append(np.float16(dcm_info.ImagePositionPatient[2]))
        filenames.append(dcm_list[sl])
    instance_uids = np.array(instance_uids)
    values, counts = np.unique(instance_uids, return_counts=True)
    dominant_uid = values[np.argmax(counts)]
    if (instance_uids != dominant_uid).sum() > 0:
        warnings.warn(f"Multiple Series UID has been detected.")
    for k in np.sort(np.argwhere(instance_uids != dominant_uid).flatten())[::-1]:
        dcm_array.pop(k)
        instance_num.pop(k)
        sl_loc.pop(k)
        dcm_infos.pop(k)
        filenames.pop(k)

    if len(set(instance_num)) < len(instance_num):
        if len(set(instance_num)) == 1:
            warnings.warn(f"Invalid instance number has been detected : Use z-coordinates instead")
            instance_num = np.argsort(sl_loc)
        else:
            warnings.warn(f"Duplicate instance number has been detected.")
            seen = set()
            for k, inum in list(enumerate(instance_num))[::-1]:
                if inum in seen:
                    dcm_array.pop(k)
                    instance_num.pop(k)
                    sl_loc.pop(k)
                    dcm_infos.pop(k)
                    filenames.pop(k)
                else:
                    seen.add(inum)

    dcm_array  = np.array(dcm_array)
    dcm_array[np.isnan(dcm_array)] = 0
    sort_idx = np.argsort(instance_num)
    dcm_array = np.array(dcm_array)[sort_idx]
    sl_loc = np.array(sl_loc)[sort_idx]
    dcm_infos = np.array(dcm_infos)[sort_idx]
    filenames = np.array(filenames)[sort_idx]

    flip_idx = sl_loc[0] > sl_loc[-1]
    if flip_idx and autoflip:
        dcm_array = np.flip(dcm_array,0)
        dcm_infos = np.flip(dcm_infos,0)
        filenames = np.flip(filenames,0)
    
    if not slice_first:
        dcm_array = np.transpose(dcm_array,(1,2,0))

    pixel_spacing = dcm_infos[0].PixelSpacing
    slice_spacing = get_slice_spacing([dcm_infos[0], dcm_infos[1]])
    thickness = get_slice_thickness([dcm_infos[0], dcm_infos[1]])

    return_array = dcm_array

    if return_metadata:
        metadata = {
            "pixel_spacing" : (abs(float(pixel_spacing[0])), abs(float(pixel_spacing[1]))),
            "slice_spacing" : abs(float(slice_spacing)),
            "thickness" : abs(float(thickness)),
            "flip_idx": flip_idx,
            "filenames": filenames
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

def get_magnetic_field_strength(dcm_path):
    dcm_info = read_dicom(os.path.join(dcm_path, os.listdir(dcm_path)[0]))
    return float(dcm_info.MagneticFieldStrength)

def get_pixel_spacing(dcm_path):
    dcm_list = sorted(os.listdir(dcm_path))
    dcm_info = read_dicom(os.path.join(dcm_path,dcm_list[0]))
    return [float(dcm_info.PixelSpacing[0]), float(dcm_info.PixelSpacing[1])]

def get_slice_spacing(dcm_path, threshold=0.5, return_both=False):
    if type(dcm_path) is str:
        dcm_list = sorted(os.listdir(dcm_path))
        dcm_info1 = read_dicom(os.path.join(dcm_path,dcm_list[0]))
        dcm_info2 = read_dicom(os.path.join(dcm_path,dcm_list[1]))
    else:
        dcm_info1 = dcm_path[0]
        dcm_info2 = dcm_path[1]
    slice_spacing_approx = abs(dcm_info1.ImagePositionPatient[2] - dcm_info2.ImagePositionPatient[2])
    if hasattr(dcm_info1,'SpacingBetweenSlices'):
        slice_spacing_field = dcm_info1.SpacingBetweenSlices
        if abs(slice_spacing_field - slice_spacing_approx) > threshold:
            slice_spacing = slice_spacing_approx
        else:
            slice_spacing = slice_spacing_field
    else:
        slice_spacing = slice_spacing_approx
    if return_both:
        return float(slice_spacing_field), float(slice_spacing_approx)
    else:
        return float(slice_spacing)

def check_slice_spacing(dcm_path, threshold=0.5):
    if type(dcm_path) is str:
        dcm_list = sorted(os.listdir(dcm_path))
        dcm_info1 = read_dicom(os.path.join(dcm_path,dcm_list[0]))
        dcm_info2 = read_dicom(os.path.join(dcm_path,dcm_list[1]))
    else:
        dcm_info1 = dcm_path[0]
        dcm_info2 = dcm_path[1]
    dcm_info1 = read_dicom(os.path.join(dcm_path,dcm_list[0]))
    dcm_info2 = read_dicom(os.path.join(dcm_path,dcm_list[1]))
    slice_spacing_approx = abs(dcm_info1.ImagePositionPatient[2] - dcm_info2.ImagePositionPatient[2])
    if hasattr(dcm_info1,'SpacingBetweenSlices'):
        slice_spacing_field = dcm_info1.SpacingBetweenSlices
        if abs(slice_spacing_field - slice_spacing_approx) > threshold:
            return True
        else:
            return False
    else:
        return False

def get_slice_thickness(dcm_path):
    if type(dcm_path) is str:
        dcm_list = sorted(os.listdir(dcm_path))
        dcm_info1 = read_dicom(os.path.join(dcm_path,dcm_list[0]))
        dcm_info2 = read_dicom(os.path.join(dcm_path,dcm_list[1]))
    else:
        dcm_info1 = dcm_path[0]
        dcm_info2 = dcm_path[1]
    if hasattr(dcm_info1, 'SliceThickness'):
        thickness = dcm_info1.SliceThickness
    elif hasattr(dcm_info1, 'SpacingBetweenSlices'):
        thickness  = dcm_info1.SpacingBetweenSlices
    else:
        thickness = get_slice_spacing(dcm_path)
    return float(thickness)

def is_reversed(dcm_path, strict=False):
    dcm_list = sorted(os.listdir(dcm_path))
    if strict:
        instance_num = []
        sl_loc = []
        for sl in range(len(dcm_list)):
            dcm_info = read_dicom(os.path.join(dcm_path,dcm_list[sl]))
            instance_num.append(dcm_info.InstanceNumber)
            sl_loc.append(np.float16(dcm_info.ImagePositionPatient[2]))

        sort_idx = np.argsort(instance_num)
        sl_loc = np.array(sl_loc)[sort_idx]

        return sl_loc[0] > sl_loc[-1]
    else:
        dcm_path1 = os.path.join(dcm_path, dcm_list[0])
        dcm_path2 = os.path.join(dcm_path, dcm_list[-1])
        return read_dicom(dcm_path1).ImagePositionPatient[2] > read_dicom(dcm_path2).ImagePositionPatient[2]
    

def find_dcm_path(root):
    root, _, file_list = list(os.walk(root))[-1]
    return root, sorted(file_list)

def find_all_dcm_path(root):
    return list(set(list(map(os.path.dirname, Path(root).rglob('*.dcm'))))) + list(set(list(map(os.path.dirname, Path(root).rglob('*.DCM')))))



def arrange_aiscan(root, remove_duplicate=False):
    t = find_all_dcm_path(root)
    temp_root = os.path.join(dn(root), bn(root + "_temp"))
    for path in t:
        patient_id = bn(dn(dn(dn(path))))
        dst = os.path.join(temp_root, patient_id)
        if not os.path.exists(dst):
            shutil.move(path, dst)
        else:
            if remove_duplicate:
                shutil.rmtree(path)
            else:
                suffix = 2
                while os.path.exists(os.path.join(temp_root, patient_id+ f"_{suffix}")):
                    suffix += 1
                dst = os.path.join(temp_root, patient_id+ f"_{suffix}")
                shutil.move(path, dst)
    shutil.rmtree(root)
    shutil.move(temp_root, root)