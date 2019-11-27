import numpy as np
import os
import random
import shutil

from sacred import Ingredient

ingredient = Ingredient('parse_months')

@ingredient.config
def cfg():
    start_at_x = 510
    cut_y = 124
    file_path = 'path_to_raw_images'

@ingredient.capture
def parse_months_run(start_at_x, cut_y):
    print(start_at_x, cut_y)
    targets = ['dmenr', 'dmer']
    for target in targets:
        # delete all files in parsed and images folder
        if os.path.exists('data/parsed/%s' % target):
            shutil.rmtree('data/parsed/%s' % target)
        if os.path.exists('data/images/%s' % target):
            shutil.rmtree('data/images/%s' % target)
        parse_target(target)

from keras_preprocessing.image import array_to_img
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split(data):
    length = data.shape[0] // 10
    splits = []

    for i in range(10):
        splits += [data[(i+0)*length:(i+1)*length]]

    return splits


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            write('E: %s' % file_path)

@ingredient.capture
def parse_file(target, traintest, file, file_path):
    start_at_x = 510
    cut_y = 124
    if traintest is not None:
        path =  '%s/%s/%s/%s' % (file_path, target, traintest, file)
    else:
        path = '%s/%s/%s' % (file_path, target, file)

    img = Image.open(path)
    w, h = img.size

    if w == 498 and h == 472:
        img.thumbnail((224, 224), Image.ANTIALIAS)
    else:
        img = img.crop((start_at_x, 0, w, h-cut_y))
        img = img.resize((498, 472))

    arr = np.array(img).mean(axis=-1)
    return arr


def check_for_split(target):
    return any([entry in ['test', 'train'] for entry in os.listdir('data/raw/%s' % target)])


def check_for_consecutive(prev_index, index):
    try:
        index = int(index)
        return prev_index + 1 == index or (index <= 3 and prev_index == index)
    except:
        return True


def to_mat(patients):
    num = sum([len(eyes) for eyes in patients])
    mat = np.zeros((num, 472, 498, 3))
    for i, eyes in enumerate(patients):
        for img in eyes:
            mat[i, :] = img

    return mat


def parse_files(target, files):
    files.sort()

    mats_train = {}
    mats_test = {}

    mat = np.zeros((2, 3, 472, 498))
    counter = 0
    prev_id = ''
    print('FILES: %d' % len(files))
    for file in files:      
        if 'DS' in file:
            continue
        
        id, eye, month, index = file.split('.',1)[0].split('_')
        counter += 1

        # print('p: %s - c: %s' %(prev_id,id))
        
        if prev_id != '' and prev_id != id: # add image
            add_image(prev_id, mat, mats_train, mats_test)
        
        # set previous data
        prev_id = id
        image = parse_file(target, None, file)
        mat[int(month) if month == '0' else 1, int(index), :, :] = image

        # add last image
        if len(files) == counter:
            add_image(prev_id, mat, mats_train, mats_test)

    print('counter: %d' % counter)
    print('train: %d' % len(mats_train))
    print('test: %d' % len(mats_test))

    return mats_train, mats_test

def add_image(id, mat, mats_train, mats_test):
    if np.random.random() < 0.8:
        mats_train[id] = mat
    else:
        mats_test[id] = mat
        mat = np.zeros((2, 3, 472, 498))
    print('add ' + id)

@ingredient.capture
def parse_target(target, file_path):
    train, test = parse_files(target, os.listdir('%s/%s' % (file_path, target)))
    save(target, train, 'train')
    save(target, test, 'test')
    

def save(target, data, folder):
    for i, d in data.items():
        path = 'data/parsed/%s/%s/%s.npy' % (target, folder, i)
        ensure_dir(path)
        np.save(path, d)

        width = 472
        height = 498

        months = np.random.randint(0, 2, (4,))
        examples = np.random.randint(0, 3, (4,))

        image = np.zeros((width, height*4, 1))
        for j, (mon, exa) in enumerate(zip(months, examples)):
            h_fr = height * j
            h_to = height * (1+j)
            w_fr = 0
            w_to = width
            image[w_fr:w_to, h_fr:h_to, 0] = d[mon, exa]

        image = array_to_img(image)
        path = 'data/images/%s/%s/%s.png' % (target, folder, i)
        ensure_dir(path)
        image.save(path)


if __name__ == '__main__':
    start_at_x = 510
    cut_y = 124

    targets = ['dmenr', 'dmer']

    for target in targets:
        parse_target(target)

