import os
import sys
import argparse
import shutil

import json
import hashlib

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="./train", type=str)
parser.add_argument("--test", default="./test", type=str)

def checkPath(pth):
    if not os.path.exists(pth):
        print(f"Not find : {pth}")
        sys.exit(1)

opt = parser.parse_args()

checkPath(opt.train)
checkPath(opt.test)

# label: train_label_path
trainLabel = {d: os.path.join(opt.train, d)  for d in os.listdir(opt.train) if os.path.isdir(os.path.join(opt.train, d))}

# generate label_names.json
print("label_names.json will be generated")
labelDict = {l:i for i, (l,ld)  in enumerate(trainLabel.items())}
with open(os.path.join(opt.train, "label_names.json"), 'w', encoding="utf-8") as f:
    json.dump(labelDict, f, ensure_ascii=False)

attr_prefix = "ss_attr"
label_prefix = ["not", "is"]

# generate label.txt
with open(os.path.join(opt.train, "label.txt"), 'w', encoding="utf-8") as f:
    for lname, idx in labelDict.items():
        f.write(f"2;{attr_prefix}_{idx};{lname}\n")
        f.write(f"{label_prefix[0]}_{attr_prefix}_{idx};{label_prefix[0]}_{attr_prefix}_{idx}\n")
        f.write(f"{label_prefix[1]}_{attr_prefix}_{idx};{label_prefix[1]}_{attr_prefix}_{idx}\n")

# generate data.txt for train
print("************* train data.txt *************")
with open(os.path.join(opt.train, "data.txt"), 'w', encoding="utf-8") as f:
    for lname, idx in labelDict.items():
        labelImgPath = trainLabel[lname]
        for fname in os.listdir(labelImgPath):
            imgPath = os.path.join(labelImgPath, fname)
            imgData = Image.open(imgPath)
            dataInfo = {
                "box": {"y":0,"x":0,"w":imgData.size[0],"h":imgData.size[1]},
                "image_id": hashlib.sha256(imgPath.encode("utf-8")).hexdigest()[:25],
                "image_file": imgPath,
                "size": {"width":imgData.size[0],"height":imgData.size[1]},
                "id": []
            }
            for i in range(len(labelDict.keys())):
                if i == idx:
                    dataInfo['id'].append(f"{label_prefix[1]}_{attr_prefix}_{i}")
                else:
                    dataInfo['id'].append(f"{label_prefix[0]}_{attr_prefix}_{i}")

            f.write(json.dumps(dataInfo, ensure_ascii=False))
            f.write("\n")
        print(f"{lname} done")

# label: train_label_path
testLabel = {d: os.path.join(opt.test, d)  for d in os.listdir(opt.test) if os.path.isdir(os.path.join(opt.test, d))}
shutil.copyfile(os.path.join(opt.train, "label.txt"), os.path.join(opt.test, "label.txt"))

# generate data.txt for test
print("************* test  data.txt *************")
with open(os.path.join(opt.test, "data.txt"), 'w', encoding="utf-8") as f:
    for lname, idx in labelDict.items():
        if lname not in trainLabel:
            continue
        labelImgPath = testLabel[lname]
        for fname in os.listdir(labelImgPath):
            imgPath = os.path.join(labelImgPath, fname)
            imgData = Image.open(imgPath)
            dataInfo = {
                "box": {"y":0,"x":0,"w":imgData.size[0],"h":imgData.size[1]},
                "image_id": hashlib.sha256(imgPath.encode("utf-8")).hexdigest()[:25],
                "image_file": imgPath,
                "size": {"width":imgData.size[0],"height":imgData.size[1]},
                "id": []
            }
            for i in range(len(labelDict.keys())):
                if i == idx:
                    dataInfo['id'].append(f"{label_prefix[1]}_{attr_prefix}_{i}")
                else:
                    dataInfo['id'].append(f"{label_prefix[0]}_{attr_prefix}_{i}")

            f.write(json.dumps(dataInfo, ensure_ascii=False))
            f.write("\n")
        print(f"{lname} done")
