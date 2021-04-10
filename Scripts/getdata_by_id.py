import sys, os, datetime, json, pycocotools

coco_id_name_map = {
1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

def main(output_dir='./json', ann_file='train.txt',rg=(0,39)):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    out_ann_file = os.path.join(output_dir, str([rg[0],rg[1]]) + '-' + ann_file.split('/')[-1])
    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='detection',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    #重新分配分类id
    new_id_to_name_map = {}
    to_new_id = {}

    for i, (id_ , class_name) in enumerate(coco_id_name_map.items()):
        if i in range(rg[0], rg[1]+1):
            new_id = i - rg[0]
            to_new_id[id_] = new_id
            new_id_to_name_map[new_id] = coco_id_name_map[id_]
            data['categories'].append(dict(
                supercategory=None,
                id=new_id,
                name=class_name,
            ))
    
    print(to_new_id)
    print(new_id_to_name_map)

    with open(ann_file, 'r', encoding='utf8') as f:
        dic = json.loads(f.read())

        #先全加进去，训练时会丢掉不用的图
        data['images'] = dic['images']

        # 找到特定类的ann
        for ann in dic['annotations']:
            cat_id = ann['category_id'] 
            if cat_id in to_new_id.keys():
                data['annotations'].append(
                    dict(
                    id=len(data['annotations']),
                    image_id=ann['image_id'],
                    category_id=to_new_id[cat_id],
                    segmentation=[[]],
                    area=ann['area'],
                    bbox=ann['bbox'],
                    iscrowd=0
                    )
                )
            # 将ann对应的图加入images
            # 暂无

        print(len(data['categories']), len(data['annotations']))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    ann_file = "../../Data/COCO2017/annotations/instances_train2017.json"
    rg = (40,49)

    main(output_dir='./json', ann_file=ann_file, rg=rg)
    print(" done!")