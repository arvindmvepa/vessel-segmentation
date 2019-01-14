import numpy as np
from itertools import groupby
import collections
from itertools import product
import shutil, errno

def find_closest_pos(positions, start_pos=(0,0)):
    min = np.inf
    min_index = -1
    start_pos = np.array(start_pos)
    for i in range(len(positions)):
        pos = positions[i]
        test = np.linalg.norm(start_pos - pos)
        if test < min:
            min = test
            min_index = i
    min_pos = positions.pop(min_index)
    return min_pos, positions


def find_class_balance(targets, masks):
    total_pos = 0
    total_num_pixels = 0
    for target, mask in zip(targets, masks):
        target = np.multiply(target, mask)
        total_pos += np.count_nonzero(target)
        total_num_pixels += np.count_nonzero(mask)
    total_neg = total_num_pixels - total_pos
    weight = total_neg / total_pos
    return weight, float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)


def remove_duplicates(data):
    ''' Remove duplicates from the data (normally a list).
        The data must be sortable and have an equality operator
    '''
    data = sorted(data)
    return [k for k, v in groupby(data)]


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


#https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
def flatten(d):
    items = []
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v).items())
        else:
            items.append((k, v))
    return dict(items)


#https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    prod_dicts = []
    for instance in product(*vals):
        prod_dicts.append(dict(zip(keys, instance)))
    return prod_dicts


#https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python
def copy_stuff(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise