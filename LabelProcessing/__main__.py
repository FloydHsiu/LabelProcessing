from optparse import OptionParser
from os import mkdir, listdir
from os.path import join, isdir, isfile
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
from LabelProcessing import LabeledImage, TransmitLabel, ClipImage
import cv2 as cv

def args_parser():
    parser = OptionParser()
    parser.add_option('-m', '--mode', dest='mode', help='write report to MODE', metavar='MODE')
    parser.add_option('-l', '--label', dest='label_dir', help='write report to LABEL_DIR', metavar='LABEL_DIR')
    parser.add_option('-i', '--image', dest='image_dir', help='wirte report to IMAGE_DIR', metavar='IMAGE_DIR')
    parser.add_option('-s', '--stride', dest='stride', help='write report to STRIDE', metavar='STRIDE')
    parser.add_option('-n', '--number', dest='num', help='write report to NUM', metavar='NUM')
    (options, args) = parser.parse_args()
    return options, args

def clip_image(image_dir, lable_dir, stride):
    clip_stride = 2
    try:
        clip_stride = int(stride)
    except Exception as e:
        pass
        
    print(f'Use Clip Stride {clip_stride} to Clip Image')
    images, labels, names = ClipImage.load_images_and_labels(image_dir, lable_dir)
    clip_dir_path = join(join(image_dir, '..'), 'clip/')
    clip_image_dir_path = join(join(image_dir, '..'), 'clip/image/')
    clip_label_dir_path = join(join(image_dir, '..'), 'clip/label/')

    #create clip image, label directory
    if not isdir(clip_dir_path): 
        mkdir(clip_dir_path)
    if not isdir(clip_image_dir_path): 
        mkdir(clip_image_dir_path)
    if not isdir(clip_label_dir_path): 
        mkdir(clip_label_dir_path)

    for i in tqdm.tqdm(range(0, len(images)), desc='Clip Images, Labels'):
        label = labels[i]
        clip_images, clip_annotations_list = ClipImage.clip_shiftx(images[i], label['annotation'], 510, 300, clip_stride)
        for j in tqdm.tqdm(range(0, len(clip_images)), desc='Save Clipped Datas'):
            label = labels[i]
            #clip data name
            clip_name = f'{names[i]}_S{clip_stride}_W300_{j+1:03d}'
            clip_image_path = join(clip_image_dir_path, f'{clip_name}.jpg')
            clip_label_path = join(clip_label_dir_path, f'{clip_name}.json')
            #update label info
            label['annotation'] = clip_annotations_list[j]
            label['image']['file_name'] = f'{clip_name}.jpg'
            label['image']['height'] = clip_images[j].shape[0]
            label['image']['width'] = clip_images[j].shape[1]
            #save datas
            cv.imwrite(clip_image_path, clip_images[j])
            with open(clip_label_path, 'w') as f:
                json.dump(label, f)

def show_image(image_dir, label_dir):
    image_path = image_dir
    label_path = label_dir
    if isfile(image_dir) and isfile(label_dir):
        with open(label_path, 'r') as f:
            label = json.load(f)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        img_seg = np.copy(img)
        img_parallelogram = np.copy(img)
        img_parallelogram_p = np.copy(img)
        img_bbox_angle= np.copy(img)
        img_bbox_center = np.copy(img)
        img_bbox_lefttop = np.copy(img)
        for a in label['annotation']:
            img_seg = LabeledImage.draw_seg(img_seg, a['segmentation'])
            img_parallelogram = LabeledImage.draw_parallelogram(img_parallelogram, a['parallelogram'])
            img_parallelogram_p = LabeledImage.draw_parallelogram_p(img_parallelogram_p, a['parallelogram_point'])
            img_bbox_angle = LabeledImage.draw_bbox_angle(img_bbox_angle, a['bbox_angle'])
            img_bbox_center = LabeledImage.draw_bbox_center(img_bbox_center, a['bbox_center'])
            img_bbox_lefttop = LabeledImage.draw_bbox_lefttop(img_bbox_lefttop, a['bbox'])
        
        _, axes = plt.subplots(2, 2, figsize=(21, 10))
        axes[0][0].set_title('bbox_center')
        axes[0][1].set_title('bbox_angle')
        axes[1][0].set_title('parallelogram')
        axes[1][1].set_title('segmentation')
        axes[0][0].imshow(img_bbox_center)
        axes[0][1].imshow(img_bbox_angle)
        axes[1][0].imshow(img_parallelogram)
        axes[1][1].imshow(img_seg)
        for ax in axes:
            for a in ax:
                #a.set_aspect('auto')
                pass
        plt.show()

def label_transmit(label_dir):
    new_label_dir = join(label_dir, 'Transmitted')
    if not isdir(new_label_dir):
        mkdir(new_label_dir)
    if isdir(label_dir):
        for n in tqdm.tqdm(listdir(label_dir), desc='Transmit Labels'):
            n_sub = n.split('.')
            n_path = join(label_dir, n)
            new_n_path = join(new_label_dir, n)
            if isfile(n_path) and n_sub[-1]=='json':
                try:
                    with open(n_path, 'r') as f:
                        label = json.load(f)
                    for i in range(0, len(label['annotation'])):
                        old_bbox = label['annotation'][i]['bbox'] #old_bbox is seg for real
                        seg = old_bbox
                        bbox_center = TransmitLabel.seg2bbox_center(seg)
                        bbox_lefttop = TransmitLabel.seg2bbox_lefttop(seg)
                        parallelogram = TransmitLabel.seg2parallelogram(seg)
                        parallelogram_point = TransmitLabel.seg2parallelogram_point(seg)
                        bbox_angle = TransmitLabel.seg2bbox_angle(seg)
                        label['annotation'][i]['segmentation'] = seg
                        label['annotation'][i]['bbox_center'] = bbox_center
                        label['annotation'][i]['bbox'] = bbox_lefttop
                        label['annotation'][i]['bbox_angle'] = bbox_angle
                        label['annotation'][i]['parallelogram'] = parallelogram
                        label['annotation'][i]['parallelogram_point'] = parallelogram_point
                        #label['annotation'][i].pop('bbox', None)
                    with open(new_n_path, 'w') as f:
                        json.dump(label, f)
                except Exception as e:
                    print(f'Error: {n_path}')

def resize(image_dir, label_dir):
    print(f'Start to Resize Image and Label')
    resized_dir = join(join(label_dir, '..'), 'resized')
    resized_img_dir = join(resized_dir, 'image')
    resized_label_dir = join(resized_dir, 'label')
    if not isdir(resized_dir):
        mkdir(resized_dir)
        if not isdir(resized_img_dir):
            mkdir(resized_img_dir)
        if not isdir(resized_label_dir):
            mkdir(resized_label_dir)
    for i in tqdm.tqdm(listdir(image_dir)):
        #parse i check if it is '.jpg'
        i_sub = i.split('.')
        if len(i_sub) == 2 and i_sub[-1] == 'jpg':
            i_path = join(image_dir, i)
            l_path = join(label_dir, f'{i_sub[0]}.json')
            new_i_path = join(resized_img_dir, i)
            new_l_path = join(resized_label_dir, f'{i_sub[0]}.json')
            if not isfile(l_path):
                break
            #load label and image
            try:
                with open(l_path, 'r') as f:
                    label = json.load(f)
                image = cv.imread(i_path)
                #calculate scale
                origin = image.shape
                if origin[0] >= origin[1]:
                    scale = 800 / origin[1]
                else:
                    scale = 800 / origin[0]
                #resize image, label
                image = cv.resize(image, (0,0), fx=scale, fy=scale)
                for i in range(len(label['annotation'])):
                    label['annotation'][i]['bbox'] = \
                    TransmitLabel.resize2short(label['annotation'][i]['bbox'], origin, 800)
                label['image']['height'] = image.shape[0]
                label['image']['width'] = image.shape[1]
                #save image, label
                with open(new_l_path, 'w') as f:
                        json.dump(label, f)
                cv.imwrite(new_i_path, image)
            except Exception as e:
                print(f'Error: Can\'t resize {i}')

def labeled_image(image_dir, label_dir, num):
    print(f'Start to Save Labeled Image')
    labeledimage_dir = join(join(image_dir, '..'), 'labeled_image')
    if not isdir(labeledimage_dir):
        mkdir(labeledimage_dir)
    ls = listdir(image_dir)
    try:
        num = int(num)
    except Exception as e:
        num = len(ls)
    for i in tqdm.tqdm(range(num)):
        m = ls[i]
        #parse i check if it is '.jpg'
        i_sub = m.split('.')
        if len(i_sub) == 2 and i_sub[-1] == 'jpg':
            i_path = join(image_dir, m)
            l_path = join(label_dir, f'{i_sub[0]}.json')
            new_i_path = join(labeledimage_dir, f'{i_sub[0]}.jpg')
            if not isfile(l_path):
                break
            with open(l_path, 'r') as f:
                label = json.load(f)
            image = cv.imread(i_path, cv.IMREAD_GRAYSCALE)
            img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            img_seg = np.copy(img)
            for a in label['annotation']:
                img_seg = LabeledImage.draw_seg(img_seg, a['segmentation'])
            cv.imwrite(new_i_path, img_seg)   

def expand(image_dir, label_dir):
    print(f'Start to Expand Label')
    expand_dir = join(join(label_dir, '..'), 'expand')
    expand_label_dir = join(expand_dir, 'label')
    if not isdir(expand_dir):
        mkdir(expand_dir)
        if not isdir(expand_label_dir):
            mkdir(expand_label_dir)
    for m in tqdm.tqdm(listdir(image_dir)):
        #parse i check if it is '.jpg'
        i_sub = m.split('.')
        if len(i_sub) == 2 and i_sub[-1] == 'jpg':
            i_path = join(image_dir, m)
            l_path = join(label_dir, f'{i_sub[0]}.json')
            new_l_path = join(expand_label_dir, f'{i_sub[0]}.json')
            if not isfile(l_path):
                break
            #load label and image
            try:
                with open(l_path, 'r') as f:
                    label = json.load(f)
                image = cv.imread(i_path)
                #calculate scale
                origin = image.shape
                #resize image, label
                new_annotation = []
                for i in range(len(label['annotation'])):
                    seg = TransmitLabel.seg2seg_expand(label['annotation'][i]['bbox'], origin, 30.0, 40.0)
                    if not seg:
                        pass
                    else:
                        label['annotation'][i]['bbox'] = seg
                        new_annotation.append(label['annotation'][i])
                label['annotation'] = new_annotation
                #save image, label
                with open(new_l_path, 'w') as f:
                        json.dump(label, f)
            except Exception as e:
                print(f'Error: Can\'t expand {m} : {e}')

def collect_labels(label_dir):
    print('Start to collect labels into one file')
    collected_path = join(join(label_dir, '..'), 'labels.json')
    if not isdir(label_dir):
        print(f'Error: {label_dir} is not directory')
        return
    collected_label = {}
    collected_label['images'] = []
    collected_label['annotations'] = []
    collected_label['categories'] = []
    image_id = 0
    annotation_id = 0
    for l in tqdm.tqdm(listdir(label_dir)):
        label_path = join(label_dir, l)
        try:
            with open(label_path, 'r') as f:
                label = json.load(f)
            image = label['image']
            image['id'] = image_id
            annotation = label['annotation']
            for i in range(len(annotation)):
                annotation[i]['id'] = annotation_id
                annotation[i]['category_id'] = 1
                annotation[i]['image_id'] = image_id
                annotation[i]['area'] = TransmitLabel.seg2area(annotation[i]['segmentation'])
                annotation[i]['iscrowd'] = 1
                annotation_id = annotation_id + 1
            collected_label['images'].append(image)
            collected_label['annotations'].extend(annotation)
            image_id = image_id + 1
        except Exception as e:
            print(f'Error: {e} \n {label_path}')
    collected_label['categories'].append({'id':1, 'name':'Flank', 'supercategory':'Endmill'})
    print(f'Totally collect: ')
    print(f"Images: {len(collected_label['images'])}")
    print(f"Annotations: {len(collected_label['annotations'])}")
    with open(collected_path, 'w') as f:
        json.dump(collected_label, f)           

def main():
    options, args = args_parser()
    mode = options.mode
    image_dir = options.image_dir
    label_dir = options.label_dir
    stride = options.stride
    num = options.num
    if mode == 'clip':
        if image_dir is not None and label_dir is not None:
            clip_image(image_dir, label_dir, stride)
    if mode == 'show':
        if image_dir is not None and label_dir is not None:
            show_image(image_dir, label_dir)
    if mode == 'transmit': #this must be done at the end
        if label_dir is not None:
            label_transmit(label_dir)
    if mode == 'resize':
        if image_dir is not None and label_dir is not None:
            resize(image_dir, label_dir)
    if mode == 'expand':
        if image_dir is not None and label_dir is not None:
            expand(image_dir, label_dir)
    if mode == 'labeledimage':
        if image_dir is not None and label_dir is not None:
            labeled_image(image_dir, label_dir, num)
    if mode == 'collectlabel':
        if label_dir is not None:
            collect_labels(label_dir)

if __name__=='__main__':
    main()