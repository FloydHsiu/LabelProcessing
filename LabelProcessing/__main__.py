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
            img_bbox_lefttop = LabeledImage.draw_bbox_lefttop(img_bbox_lefttop, a['bbox_lefttop'])
        
        _, axes = plt.subplots(1, 4, figsize=(21, 5))
        axes[0].set_title('bbox_center')
        axes[1].set_title('bbox_angle')
        axes[2].set_title('parallelogram')
        axes[3].set_title('segmentation')
        axes[0].imshow(img_bbox_center)
        axes[1].imshow(img_bbox_angle)
        axes[2].imshow(img_parallelogram)
        axes[3].imshow(img_seg)
        for ax in axes:
            #ax.set_aspect('auto')
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
                        label['annotation'][i]['bbox_lefttop'] = bbox_lefttop
                        label['annotation'][i]['bbox_angle'] = bbox_angle
                        label['annotation'][i]['parallelogram'] = parallelogram
                        label['annotation'][i]['parallelogram_point'] = parallelogram_point
                        label['annotation'][i].pop('bbox', None)
                    with open(new_n_path, 'w') as f:
                        json.dump(label, f)
                except Exception as e:
                    print(f'Error: {n_path}')

def main():
    options, args = args_parser()
    mode = options.mode
    image_dir = options.image_dir
    label_dir = options.label_dir
    stride = options.stride
    if mode == 'clip':
        if image_dir is not None and label_dir is not None:
            clip_image(image_dir, label_dir, stride)
    if mode == 'show':
        if image_dir is not None and label_dir is not None:
            show_image(image_dir, label_dir)
    if mode == 'transmit':
        if label_dir is not None:
            label_transmit(label_dir)

if __name__=='__main__':
    main()