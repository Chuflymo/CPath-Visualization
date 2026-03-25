import numpy as np
import xml
import openslide
import h5py
import torch
import os
import xml.dom.minidom

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'


def get_cam_1d(classifier, feat, attention):
    try:
        if len(attention.shape) > 1:
            n, m = feat.shape
            dim1, dim2 = attention.shape
            if dim1 != n:
                attention = attention.T
            dim1, dim2 = attention.shape
            feat = feat.reshape(n, dim2, m // dim2)
            features = torch.einsum('nbs,nb->nbs', feat, attention)
            features = features.reshape(n, m)
        else:
            features = torch.einsum('nm,n->nm', feat, attention)
        tweight = list(classifier.parameters())[-2]
        cam_maps = torch.einsum('gf,cf->cg', features, tweight)
        return cam_maps
    except Exception as e:
        print(f"get_cam_1d error: {str(e)}")
        if hasattr(feat, 'shape'):
            print(f"feat.shape error: {feat.shape}")
        return None


def read_annotation(anno_file, return_type=False):
    anno_tumor = []
    anno_normal = []
    anno_type = set()
    DOMTree = xml.dom.minidom.parse(anno_file)
    annotations = DOMTree.documentElement.getElementsByTagName('Annotations')[0].getElementsByTagName('Annotation')
    for i in range(len(annotations)):
        anno_type.add(annotations[i].getAttribute('PartOfGroup'))
        if annotations[i].getAttribute('PartOfGroup') == 'Exclusion':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])
            anno_normal.append(_tmp)
        elif annotations[i].getAttribute('PartOfGroup') != 'None':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])
            anno_tumor.append(_tmp)
    if return_type:
        return anno_tumor, anno_normal, anno_type
    else:
        return anno_tumor, anno_normal


def get_area(pos_anchors, margin_percentage=2.0, center_anchors=None, width_height=None):
    if center_anchors is not None:
        margin = 10 * 512 * margin_percentage / 2
        center_anchor = (center_anchors[1], center_anchors[0])
    else:
        top, down, left, right = min(pos_anchors[:, 1]), max(pos_anchors[:, 1]), min(pos_anchors[:, 0]), max(pos_anchors[:, 0])
        center_anchor = ((top + down) // 2, (left + right) // 2)
        margin = max((down - top), (right - left)) * margin_percentage / 2

    top, down, left, right = np.array(
        [center_anchor[0] - margin, center_anchor[0] + margin, center_anchor[1] - margin, center_anchor[1] + margin],
        dtype=int
    )
    ori_coord = np.array([down, top, right, left])
    top, down, left, right = np.clip(top, 0, width_height[1]), np.clip(down, 0, width_height[1]), np.clip(left, 0, width_height[0]), np.clip(right, 0, width_height[0])
    _gap = np.array([down, top, right, left])
    _coord = np.array([top, down, left, right])
    top, down, left, right = _coord + (_gap - ori_coord)

    return top, down, left, right, (down - top) * (right - left)


def screen_coords(scores, coords, top_left, bot_right, cam=None):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    if cam is not None:
        try:
            return scores[mask], coords[mask], cam[mask]
        except Exception as e:
            return None, None, None
    else:
        try:
            return scores[mask], coords[mask]
        except Exception as e:
            return None, None


def generate_heatmap(tif_path, xml_path, h5_path, attention_path,
                     vis_level=3, alpha=0.3, margin_percentage=1.5,
                     filter_thr=0.3, rel_roi_idx=None, patch_size=512,
                     multihead_agg='mean'):

    data = torch.load(attention_path)
    h5 = h5py.File(h5_path, "r")
    loc = h5['coords']

    data = data.squeeze().cpu().numpy()
    if len(data.shape) > 1:
        if multihead_agg == 'mean':
            data = data.mean(axis=0)
        elif multihead_agg == 'max':
            data = data.max(axis=0)
        else:
            data = data.mean(axis=0)

    slide = openslide.OpenSlide(tif_path)
    width, height = slide.dimensions

    anno_tumor, anno_normal = read_annotation(xml_path)

    patch_num = 1
    roi = 0
    roi_num = 0
    if rel_roi_idx is not None:
        roi = rel_roi_idx
    else:
        for _pos in range(len(anno_tumor)):
            _, _, _, _, area = get_area(np.array(anno_tumor[_pos]), width_height=[width, height])
            if area >= (patch_size * patch_num) ** 2:
                if area ** 0.5 / patch_size >= roi_num:
                    roi = _pos
                    roi_num = area ** 0.5 / patch_size

    crop_size = 512
    stride = 512
    pos_anchors = np.array(anno_tumor[roi])
    top, down, left, right, _ = get_area(pos_anchors, margin_percentage, width_height=[width, height])

    crop_coords = []
    for i in range(top, down, stride):
        for j in range(left, right, stride):
            if j + crop_size > width or i + crop_size > height:
                continue
            crop_coords.append((j, i))
    right, down = np.max(crop_coords, 0) + crop_size

    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w, _h = np.array((right - left, down - top)) * scale_level_ratio
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w), int(_h))).convert('RGB')

    A_roi, coords_roi = screen_coords(data, loc, (left, top), (right, down))
    if A_roi is None or coords_roi is None:
        return None, None

    A_roi = A_roi * 1.0
    img = np.array(region)

    coords_roi_rel = []
    for i in range(len(coords_roi)):
        rel_x = coords_roi[i][0] - left
        rel_y = coords_roi[i][1] - top
        rel_x, rel_y = np.clip(rel_x, 0, right - left), np.clip(rel_y, 0, down - top)
        coords_roi_rel.append([rel_x, rel_y])

    data_min = data.min()
    data_max = data.max()
    A_roi = (A_roi - data_min) / (data_max - data_min)
    A_roi[A_roi < filter_thr] = 0

    heatmap_attn = np.zeros(img.shape)
    for _idx, _coord in enumerate(coords_roi_rel):
        heatmap_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
        int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = A_roi[_idx] * 255

    blended_image = (alpha * img + (1 - alpha) * heatmap_attn[:, :, :3])
    blended_image = blended_image.astype(np.uint8)

    from PIL import ImageDraw, Image
    pil_img = Image.fromarray(blended_image)
    draw = ImageDraw.Draw(pil_img)
    scale_ratio = scale_level_ratio

    for anchors in anno_tumor:
        _pos_anchors = np.array(anchors)
        coords_list = [((x - left) * scale_ratio[0], (y - top) * scale_ratio[1]) for x, y in _pos_anchors]
        draw.line(coords_list, fill='deepskyblue', width=2)

    return np.array(pil_img), roi


def generate_cam(tif_path, xml_path, h5_path, attention_path,
                 feat_path, classifier_path,
                 vis_level=3, alpha=0.3, margin_percentage=1.5,
                 filter_thr=0.2, filter_thr_cam=0.3,
                 rel_roi_idx=None, patch_size=512, cam_norm_method='softmax', cam_class_idx=0):


    data = torch.load(attention_path)
    feat = torch.load(feat_path)
    classifier = torch.load(classifier_path)

    h5 = h5py.File(h5_path, "r")
    loc = h5['coords']

    data = data.squeeze()
    feat = feat.squeeze()

    cam = get_cam_1d(classifier, feat, data)
    if cam is None:
        return None, None

    data = data.squeeze().cpu().numpy()
    feat = feat.squeeze().cpu().numpy()

    if cam_norm_method == 'Min-Max':
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-10)
        cam = cam[cam_class_idx]
    else:
        cam = torch.nn.functional.softmax(cam, dim=0)
        cam = cam[cam_class_idx]

    cam = cam.cpu().detach().numpy()

    if len(data.shape) > 1:
            data = data.mean(axis=0)

    slide = openslide.OpenSlide(tif_path)
    width, height = slide.dimensions

    anno_tumor, anno_normal = read_annotation(xml_path)

    patch_num = 1
    roi = 0
    roi_num = 0
    if rel_roi_idx is not None:
        roi = rel_roi_idx
    else:
        for _pos in range(len(anno_tumor)):
            _, _, _, _, area = get_area(np.array(anno_tumor[_pos]), width_height=[width, height])
            if area >= (patch_size * patch_num) ** 2:
                if area ** 0.5 / patch_size >= roi_num:
                    roi = _pos
                    roi_num = area ** 0.5 / patch_size

    crop_size = 512
    stride = 512
    pos_anchors = np.array(anno_tumor[roi])
    top, down, left, right, _ = get_area(pos_anchors, margin_percentage, width_height=[width, height])

    crop_coords = []
    for i in range(top, down, stride):
        for j in range(left, right, stride):
            if j + crop_size > width or i + crop_size > height:
                continue
            crop_coords.append((j, i))
    right, down = np.max(crop_coords, 0) + crop_size

    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w, _h = np.array((right - left, down - top)) * scale_level_ratio
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w), int(_h))).convert('RGB')

    A_roi, coords_roi, cam_roi = screen_coords(data, loc, (left, top), (right, down), cam=cam)
    if A_roi is None or coords_roi is None:
        return None, None

    A_roi = A_roi * 1.0

    cam_min = cam_roi.min()
    cam_max = cam_roi.max()
    cam_roi = (cam_roi - cam_min) / (cam_max - cam_min)
    cam_roi = cam_roi * 1
    cam_roi = 1 - cam_roi

    img = np.array(region)

    coords_roi_rel = []
    for i in range(len(coords_roi)):
        rel_x = coords_roi[i][0] - left
        rel_y = coords_roi[i][1] - top
        rel_x, rel_y = np.clip(rel_x, 0, right - left), np.clip(rel_y, 0, down - top)
        coords_roi_rel.append([rel_x, rel_y])

    data_min = data.min()
    data_max = data.max()
    A_roi = (A_roi - data_min) / (data_max - data_min)
    A_roi[A_roi < filter_thr] = 0
    cam_roi[cam_roi <= filter_thr_cam] = 0

    heatmap_attn = np.zeros(img.shape)
    heatmap_cam = np.zeros(img.shape)
    alpha_mat_attn = np.ones(img.shape) * 0.5 * alpha
    alpha_mat_cam = np.ones(img.shape) * 0.5 * alpha

    _c_cam = np.array([0, 255, 255])
    _c_attn = np.array([255, 255, 255])

    for _idx, _coord in enumerate(coords_roi_rel):
        A_flag = False
        cam_flag = False

        if A_roi[_idx] != 0:
            heatmap_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = A_roi[_idx] * _c_attn
            A_flag = True

        if cam_roi[_idx] != 0:
            alpha_mat_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = 0
            heatmap_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = 0
            alpha_mat_cam[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = 1 - cam_roi[_idx]

            heatmap_cam[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = _c_cam
            cam_flag = True
        else:
            alpha_mat_cam[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = 0

        if not A_flag and not cam_flag:
            alpha_mat_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
            int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = alpha

    blended_image = (alpha_mat_attn * img + alpha_mat_cam * img +
                    (1 - alpha_mat_attn) * heatmap_attn + (1 - alpha_mat_cam) * heatmap_cam)
    blended_image = blended_image.astype(np.uint8)

    from PIL import ImageDraw, Image
    pil_img = Image.fromarray(blended_image)
    draw = ImageDraw.Draw(pil_img)
    scale_ratio = scale_level_ratio

    for anchors in anno_tumor:
        _pos_anchors = np.array(anchors)
        coords_list = [((x - left) * scale_ratio[0], (y - top) * scale_ratio[1]) for x, y in _pos_anchors]
        draw.line(coords_list, fill='deepskyblue', width=2)

    return np.array(pil_img), roi
