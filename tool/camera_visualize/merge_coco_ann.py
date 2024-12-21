"""
合并多个coco格式的json文件
"""

from ast import arg
import os
import json
# import shutil
from tqdm import tqdm
import pathlib
import argparse


def _read_jsonfile(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def _save_json(instance, save_path):
    """ 保存 coco json文件
    """
    json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

def merge_coco_ann(src_json_dir, dst_json_path, recursively=False, id_reassign = False):  
    """
    合并json数据格式
    
    备注：
        1. 默认认为两个数据集中的图片不一样
        2. 目前未处理categories不一致的情况，后续需要增加对categories的一致性验证
    """  
    if not os.path.exists(os.path.dirname(dst_json_path)):
        os.makedirs(os.path.dirname(dst_json_path))
    
    json_queue_paths=[]
    src_json_dir_path = pathlib.Path(src_json_dir)
    if recursively==True:
        json_queue_paths.append(src_json_dir_path.rglob("*.json"))
    else:
        json_queue_paths.append(src_json_dir_path.glob("*.json"))


    ann_paths = []
    for json_queue_path in tqdm(json_queue_paths):
        for json_path in tqdm(json_queue_path):
            if json_path.is_dir():
                continue
            json_full_name = os.path.normpath(str(json_path))            
            ann_paths.append(json_full_name)
    
    keypoints = {}
    keypoints['info'] = {'description': 'Lableme Dataset', 'version': 1.0, 'year': 2022}
    keypoints['license'] = [ 
        {
            "url": "http://zienon.com/",
            "id": None,
            "name": 'zienon'
        } ]
    keypoints['images'] = []
    keypoints['annotations'] = []
    keypoints['categories'] = []
    
    image_id = 0
    ann_id = 0
    for json_file_name in  tqdm(ann_paths):
        if not os.path.exists(json_file_name):
            continue
        json_data = _read_jsonfile(json_file_name)
        
        if id_reassign: # 重新分配ID
            image_id = len(keypoints['images'])
            ann_id = len(keypoints['annotations'])
            image_old_new_id_convert_map = {}
            for image in json_data['images']:
                image_old_new_id_convert_map[image['id']] = image_id
                image['id'] = image_id
                image_id += 1

            for annotation in json_data['annotations']:
                annotation['image_id'] = image_old_new_id_convert_map[annotation['image_id']]
                annotation['id'] = ann_id
                ann_id += 1
        
        keypoints['images'].extend(json_data['images'])
        keypoints['annotations'].extend(json_data['annotations'])
        keypoints['categories'] = json_data['categories']

    if len(ann_paths)>0:
        _save_json(keypoints, dst_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_json_dir", type=str, help="dataset path input", \
        default="D:/datasets/hands/handpose_datasets/ann/"
    )
    parser.add_argument(
        "--output_json", type=str, help="output annotations file name",
        default="D:/datasets/hands/handpose_datasets/ann/annotations.json"
    )
    parser.add_argument("--recursively", action='store_true', default=False, help="recursively directory")
    parser.add_argument("--id_reassign", action='store_true', default=True, help="reassign image id and annotation id")
    args = parser.parse_args()
    merge_coco_ann(src_json_dir=args.src_json_dir, dst_json_path=args.output_json, recursively=args.recursively, id_reassign=args.id_reassign)
    print("finished")
    