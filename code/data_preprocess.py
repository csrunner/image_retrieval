# -*- coding:utf-8 -*-
__author__ = 'shichao'


import cv2
import numpy as np
from skimage import exposure
import os
import glob
import random
import shutil


def data_augmentation_in_seperate_folders(img_path,is_save):
    augmentation_method = {
        'add-logo':add_logo,
        'black-edge':black_edge,
        'resize':resize,
        'rotate-by-90':rotate_by_90,
        'rotate-by-anyangle':rotate_by_anyangle,
        'affine-transform':affine_transform,
        'gamma-transform':gamma_transform,
        'blurr-transform':blurr_transform

    }
    pool = ['add-logo', 'black-edge', 'resize', 'rotate-by-90', 'rotate-by-anyangle', 'affine-transform',
            'gamma-transform', 'blurr-transform']
    sampling = np.random.choice(pool, 7, replace=False)
    train_list = sampling[:4]
    val_list = sampling[4:6]
    test_list = sampling[6:]
    saving_original_img = True
    if is_save:
        # path_188 = '/home/shichao/image_retrieval/data/train_all'
        # path_99 = '/home/data/img_25g/train/train_data'
        path_188 = '/home/shichao/image_retrieval/data/data_25g'
        path_99 = '/home/shichao/mount-dir/data/image_retrieval/data_25g_seperate'
        data_root = path_99

        train_path = os.path.join(data_root,'train')
        val_path = os.path.join(data_root,'val')
        test_path = os.path.join(data_root,'test')
        try:
            os.mkdir(train_path)
            os.mkdir(val_path)
            os.mkdir(test_path)
        except OSError:
            print('dir already existed')
        for method in train_list:

            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]

            saving_dir = os.path.join(train_path,img_name)
            try:
                os.mkdir(saving_dir)
            except OSError:
                print('sub dir already existed')
            saving_path = os.path.join(saving_dir,img_name+'_'+method+img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path,aug_img)
            if saving_original_img:
                shutil.copyfile(img_path,os.path.join(saving_dir,os.path.basename(img_path)))
                saving_original_img = False

        for method in val_list:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]

            saving_dir = os.path.join(val_path, img_name)
            try:
                os.mkdir(saving_dir)
            except OSError:
                print('sub dir already existed')
            saving_path = os.path.join(saving_dir, img_name + '_' + method + img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path, aug_img)

        for method in test_list:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]

            saving_dir = os.path.join(test_path, img_name)
            try:
                os.mkdir(saving_dir)
            except OSError:
                print('sub dir already existed')
            saving_path = os.path.join(saving_dir, img_name + '_' + method + img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path, aug_img)


def data_augmentation_in_one_folder(img_path,is_save):
    augmentation_method = {
        'add-logo':add_logo,
        'black-edge':black_edge,
        'resize':resize,
        'rotate-by-90':rotate_by_90,
        'rotate-by-anyangle':rotate_by_anyangle,
        'affine-transform':affine_transform,
        'gamma-transform':gamma_transform,
        'blurr-transform':blurr_transform

    }
    pool = ['add-logo', 'black-edge', 'resize', 'rotate-by-90', 'rotate-by-anyangle', 'affine-transform',
            'gamma-transform', 'blurr-transform']
    sampling = np.random.choice(pool, 7, replace=False)
    train_list = sampling[:4]
    val_list = sampling[4:6]
    test_list = sampling[6:]
    saving_original_img = True
    if is_save:
        # path_188 = '/home/shichao/image_retrieval/data/train_all'
        # path_99 = '/home/data/img_25g/train/train_data'
        path_188 = '/home/shichao/image_retrieval/data/data_25g'
        path_99 = '/home/shichao/mount-dir/data/image_retrieval/data_25g'
        data_root = path_99

        train_path = os.path.join(data_root,'train')
        val_path = os.path.join(data_root,'val')
        test_path = os.path.join(data_root,'test')
        try:
            os.mkdir(train_path)
            os.mkdir(val_path)
            os.mkdir(test_path)
        except OSError:
            print('dir already existed')
        for method in train_list:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]
            saving_path = os.path.join(train_path,img_name+'_'+method+img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path,aug_img)

            if saving_original_img:
                shutil.copyfile(img_path,os.path.join(saving_path,os.path.basename(img_path)))
                saving_original_img = False

        for method in val_list:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]
            saving_path = os.path.join(val_path, img_name + '_' + method  + img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path, aug_img)

        for method in test_list:
            img = cv2.imread(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(os.path.basename(img_path))[1]
            saving_path = os.path.join(test_path, img_name + '_' + method  + img_ext)

            aug_func = augmentation_method[method]
            aug_img = aug_func(img)
            cv2.imwrite(saving_path, aug_img)


def add_logo(img):  # logo is in center , need to adjust
    logo_path = '/home/shichao/mount-dir/model_code/image_retrieval/code/logo1.jpg'
    # img = cv2.imread(img_path)
    logo = cv2.imread(logo_path)
    logo = cv2.resize(logo,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)
    log_rows, logo_cols = logo.shape[:2]
    img_rows, img_cols = img.shape[:2]
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(logo_gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    r1 = 0
    c1 = 0
    r2 = r1 + log_rows
    c2 = c1 + logo_cols
    roi = img[r1:r2, c1:c2]
    img_bg = cv2.bitwise_and(roi, roi, mask = mask)
    logo_fg = cv2.bitwise_and(logo, logo, mask = mask_inv)
    dst = cv2.add(img_bg, logo_fg)
    img[r1:r2, c1:c2] = dst
    return img

'''
def black_edge(img_path,output_path): # black edage is 15% of image
    img1 = Image.open(img_path)
    img1_w, img1_h = img1.size
    img2 = Image.new("RGB", (img1_w + 2 * int(img1_w/10), img1_h + 2 * int(img1_h/10)))
    img2.paste(img1,(int(img1_w/10),int(img1_h/10)))
    img2.save(output_path)
'''



def resize(img):
    height = 480
    width = 640
    img2 = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
    # new_x, new_y = new_size
    # img = cv2.imread(img_path)
    # img2 = cv2.resize(img,(0,0),fx = new_x, fy = new_y, interpolation = cv2.INTER_NEAREST)
    #cv2.imwrite(output_path, img2)
    return img2

def rotate_by_90(img): # issue is  img just rorate b*90 angle
    # img = cv2.imread(img_path)
    r_angle = random.randint(0, 2)
    img2 = cv2.rotate(img,r_angle) # rorate 90
    return img2


def rotate_by_anyangle(img): # issue is img will be cutted
    r_angle = 360 * random.random()
    # img = cv2.imread(img_path)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, r_angle, 1)
    img2 = cv2.warpAffine(img, M, (w, h))
    #cv2.imwrite(output_path, img2)
    return img2


def affine_transform(img):
    # img = cv2.imread(img_path)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
    M = cv2.getAffineTransform(pts1, pts2)
    img2 = cv2.warpAffine(img, M, (cols, rows))
    #cv2.imwrite(output_path, img2)
    return img2

def gamma_transform(img):  # >1 : dark  <1 : light
    gam_value = 2 * random.random()
    # img = cv2.imread(img_path)
    img2 = exposure.adjust_gamma(img, gam_value)
    #cv2.imwrite(output_path, img2)
    return img2

def blurr_transform(img):
    # img = cv2.imread(img_path)
    img2 = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imwrite(output_path, img2)
    return img2

def black_edge(img):
    height = 480
    width = 640
    ratio = 0.08
    center_height = int(height*(1-2*ratio))
    center_width = int(width*(1-2*ratio))
    left = int(width*ratio)
    top = int(height*ratio)
    black = np.zeros((height,width,3))
    resized_img = cv2.resize(img,(center_width,center_height),interpolation=cv2.INTER_CUBIC)
    black[top:top+center_height,left:left+center_width,:] = resized_img
    return black



def main():
    '''

    :return:
    '''
    '''
    path = '/Users/shichao/Downloads/1.jpg'
    img = cv2.imread(path)
    black = black_edge(img)
    cv2.imwrite('/Users/shichao/Downloads/black.jpg',black)
    '''
    root = '/home/data/img_25g/train/train_data'
    files_path = glob.glob(os.path.join(root,'*.jpg'))
    for file_path in files_path:
        data_augmentation_in_seperate_folders(file_path,is_save=True)
        print(file_path)



if __name__ == '__main__':
    main()