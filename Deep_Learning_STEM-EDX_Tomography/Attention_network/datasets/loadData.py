import numpy as np

from PIL import Image
import os
import os.path
import random
import _pickle as pickle


IMG_EXTENSIONS = [ '.mat']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_subjects(dir):
    subjects = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    subjects.sort()
    subject_to_idx = {subjects[i]: i for i in range(len(subjects))}
    return subjects, subject_to_idx

def make_dataset(dir, subject_to_idx):
    ddir = os.path.join(dir, 'Label')
    images = []
    for subject in os.listdir(ddir):
        d = os.path.join(ddir, subject)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):	    
                if is_image_file(fname):
                    path_label = os.path.join(root, fname)
                    path_input = path_label.replace(ddir, os.path.join(dir, 'Input'))
                    item = (path_input, path_label)
                    images.append(item)
    return images

def get_data_idx(root):
    subjects, subject_to_idx = find_subjects(root)
    imgs = make_dataset(root, subject_to_idx)
    if len(imgs) == 0:
        raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    # split data: get indices
    split_indices_filename = os.path.join(root, 'split_indices.pkl')
    if os.path.isfile(split_indices_filename):
        with  open(split_indices_filename, 'rb') as f:
            indices = pickle.load(f)
            print('Load split indices.')
    else:
        num_data = len(imgs)
        indices = np.random.permutation(num_data)
        with  open(split_indices_filename, 'wb') as f:
            pickle.dump(indices, f)
    print('Save split indices.')

    # split data: get split data
    data = [imgs[datum] for datum in indices]
    
    return data
	
	
