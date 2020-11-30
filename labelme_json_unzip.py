#%%
import uuid
import argparse
import json
import os
import os.path as osp
import subprocess
import numpy as np
import PIL.Image
import yaml
import cv2
import yaml
import sys
from labelme import utils
import multiprocessing
# %%
def labelme_shapes_to_label(img_shape, shapes):
    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _ = utils.shapes_to_label(img_shape, shapes, label_name_to_value)
    return lbl, label_name_to_value

def shapes_to_label(img_shape, shapes):
    label_name_to_value = [("_background_", 0)]
    ins = np.zeros(img_shape[:2], dtype=np.int32)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        label_name_to_value.append((cls_name, ins_id))

        mask = utils.shape_to_mask(img_shape[:2], points, shape_type)
        ins[mask] = ins_id

    return ins, label_name_to_value

def json_unzip(path, error_files):
    assert osp.isfile(path) and path.endswith(".json"), "Check input file."
    try:
        data = json.load(open(path))
    except Exception as e:
        error_files.append((path, e))
    
    out_dir = osp.basename(path).replace('.', '_')
    out_dir = osp.join(osp.dirname(path), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    img = utils.img_b64_to_arr(data['imageData'])

    lbl_sema, lbl_sema_names = labelme_shapes_to_label(img.shape, data['shapes'])
    captions = ['%s' % (name) for name in lbl_sema_names]
    lbl_sema_viz = utils.draw_label(lbl_sema, img, captions)
    
    lbl_ins, lbl_ins_names = shapes_to_label(img.shape, data['shapes'])
    captions = ['%s' % (name) for name, _ in lbl_ins_names]
    lbl_ins_viz = utils.draw_label(lbl_ins, img, captions)
    
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))

    label_sema_path = osp.join(out_dir, 'label_sema.png')
    lbl_sema_8u = lbl_sema.copy().astype(np.uint8)
    PIL.Image.fromarray(lbl_sema_8u).save(label_sema_path)
    PIL.Image.fromarray(lbl_sema_viz).save(osp.join(out_dir, 'label_sema_viz.png'))

    label_ins_path = osp.join(out_dir, 'label_ins.png')
    lbl_ins_8u = lbl_ins.copy().astype(np.uint8)
    PIL.Image.fromarray(lbl_ins_8u).save(label_ins_path)
    PIL.Image.fromarray(lbl_ins_viz).save(osp.join(out_dir, 'label_ins_viz.png'))

    with open(osp.join(out_dir, 'label_sema_names.txt'), 'w') as f:
        for lbl_name in lbl_sema_names:
            f.write(lbl_name + '\n')
    
    with open(osp.join(out_dir, 'label_ins_names.txt'), 'w') as f:
        for lbl_name in lbl_ins_names:
            f.write(lbl_name[0] + '\n')

    print('Saved to: %s' % out_dir)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unzip json file create by labelme.')
    parser.add_argument('folder', type=str, help="Path of json files floder.")
    args = parser.parse_args()
    json_folder = args.folder
    assert osp.isdir(json_folder), "Check json floder path."
    json_files = []
    error_files = multiprocessing.Manager().list()
    for root, dirs, files in os.walk(json_folder, topdown=True):
        for name in files:
            json_files.append(os.path.join(root, name))
    json_files = [x for x in json_files if osp.split(x)[-1].endswith(".json")]

    cpu_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_num)
    for path in json_files:
        pool.apply_async(json_unzip, args=(path, error_files))
    pool.close()
    pool.join()
    print(error_files)