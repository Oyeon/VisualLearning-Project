import json

with open('ai_editor/annotations/instances_train2017.json','r') as f:
    ff = json.load(f)
    with open('ai_editor/no_mask.txt','r') as no_mask:
        no_masks = no_mask.readline().rstrip().split(',')
    print(len(no_masks))

    image_list = ff['images']
    image_dict = {img['id']:img for img in image_list}
    for id in no_masks:
        del image_dict[int(id)]


    new_image_list = list(image_dict.values())

    ff['images'] = new_image_list

    with open('ai_editor/annotations/instances_train2017_adj.json','w') as fff:
        json.dump(ff, fff)