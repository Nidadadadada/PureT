import json
#生成misc/ids2path_json/coco_train_ids2path.json等
# with open('./mscoco/misc/ids2path_json/coco_test_ids2path.json') as f:
#     test=json.load(f)
train_dict={}
val_dict={}
test_dict={}
unsupervised_dict={}
import os
imagelist_train=os.listdir(r'./MOCS/feature/coco2014/train_val_shuffle')
imagelist_val=os.listdir(r'./MOCS/feature/coco2014/val2014_shuffle')
imagelist_test=os.listdir(r'./MOCS/feature/coco2014/test2014_shuffle')
imagelist_buchong=os.listdir(r'./MOCS/feature/coco2014/buchong')
# imagelist_unsupervised=os.listdir(r'./MOCS/feature/coco2014/unsupervised')
for image in imagelist_train:
    imageid=str(int(image.split('.')[0]))
    train_dict[imageid]='train_val_shuffle/'+image
for image in imagelist_buchong:
    imageid=str(int(image.split('.')[0]))
    train_dict[imageid]='buchong/'+image

# for image in imagelist_val:
#     imageid=str(int(image.split('.')[0]))
#     val_dict[imageid]='val2014_shuffle/'+image
# for image in imagelist_test:
#     imageid=str(int(image.split('.')[0]))
#     test_dict[imageid]='test2014_shuffle/'+image

# for image in imagelist_unsupervised:
#     imageid=str(int(image.split('.')[0]))
#     unsupervised_dict[imageid]='unsupervised/'+image

print(len(train_dict))
# print(len(val_dict))
# print(len(test_dict))
json_str = json.dumps(train_dict)
# with open('./MOCS/misc/ids2path_json/train_val_buchong_ids2path.json', 'w') as json_file:
#     json_file.write(json_str)
# json_str = json.dumps(val_dict)
# with open('./MOCS/misc/ids2path_json/coco_val_ids2path_shuffle.json', 'w') as json_file:
#     json_file.write(json_str)
# json_str = json.dumps(test_dict)
# with open('./MOCS/misc/ids2path_json/coco_test_ids2path_shuffle.json', 'w') as json_file:
#     json_file.write(json_str)

# json_str = json.dumps(unsupervised_dict)
# with open('./MOCS/misc/ids2path_json/coco_train_ids2path_unsupervised.json', 'w') as json_file:
#     json_file.write(json_str)