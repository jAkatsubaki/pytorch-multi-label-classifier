import os
import sys
import argparse

import json
import hashlib

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="./", type=str)

def checkPath(pth):
    if not os.path.exists(pth):
        print(f"Not find : {pth}")
        sys.exit(1)

opt = parser.parse_args()

rootdir = opt.dir
checkPath(rootdir)

filelInDir = os.listdir(rootdir)
labelInDir = [l for l in filelInDir if os.path.isdir(os.path.join(rootdir, l))]

print("label_names.json will be generated")
labelDict = {l:i for i,l in enumerate(labelInDir)}
with open(os.path.join(rootdir, "label_names.json"), 'w', encoding="utf-8") as f:
    json.dump(labelDict, f, ensure_ascii=False)

attr_prefix = "ss_attr"
label_prefix = ["not", "is"]

with open(os.path.join(rootdir, "label.txt"), 'w', encoding="utf-8") as f:
    for lname, idx in labelDict.items():
        f.write(f"2;{attr_prefix}_{idx};{lname}\n")
        f.write(f"{label_prefix[0]}_{attr_prefix}_{idx};{label_prefix[0]}_{attr_prefix}_{idx}\n")
        f.write(f"{label_prefix[1]}_{attr_prefix}_{idx};{label_prefix[1]}_{attr_prefix}_{idx}\n")

with open(os.path.join(rootdir, "data.txt"), 'w', encoding="utf-8") as f:
    for lname, idx in labelDict.items():
        labelRootDir = os.path.join(rootdir, lname)
        for fname in os.listdir(labelRootDir):
            imgPath = os.path.join(labelRootDir, fname)
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
