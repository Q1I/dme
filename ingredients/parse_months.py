import numpy as np
import os
import random

from sacred import Ingredient

ingredient = Ingredient('parse_months')

@ingredient.config
def cfg():
    start_at_x = 510
    cut_y = 124
    file_path = '/home/q1/Python/dl/data/uniklinik_augen'

@ingredient.capture
def parse_months_run(start_at_x, cut_y):
    print(start_at_x, cut_y)
    targets = ['dmenr', 'dmer']
    for target in targets:
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

    mats_train = []
    mats_test = []

    mat = np.zeros((2, 3, 472, 498))
    index = 0

    print('FILES: %d' % len(files))
    for file in files:      
        if 'DS' in file:
            continue
        
        image = parse_file(target, None, file)

        if index % 6 < 3:
            month = 0
        else:
            month = 1
        image_index = index % 3

        mat[month, image_index, :, :] = image

        index += 1

        if index % 6 == 0:
            if np.random.random() < 0.8:
                mats_train += [mat]
            else:
                mats_test += [mat]

            mat = np.zeros((2, 3, 472, 498))

    print('index: %d' % index)
    print('train: %d' % len(mats_train))
    print('test: %d' % len(mats_test))

    return mats_train, mats_test

@ingredient.capture
def parse_target(target, file_path):
    train, test = parse_files(target, os.listdir('%s/%s' % (file_path, target)))
    save(target, train, 'train')
    save(target, test, 'test')
    

def save(target, data, folder):
    for i, d in enumerate(data):
        path = 'data/parsed/%s/%s/%i.npy' % (target, folder, i)
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
        path = 'data/images/%s/%s/%i.png' % (target, folder, i)
        ensure_dir(path)
        image.save(path)


if __name__ == '__main__':
    start_at_x = 510
    cut_y = 124

    targets = ['dmenr', 'dmer']

    for target in targets:
        parse_target(target)
