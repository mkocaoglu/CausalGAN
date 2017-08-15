"""
Modification of
https://github.com/carpedm20/BEGAN-tensorflow/blob/master/download.py
"""
from __future__ import print_function
import os
import zipfile
import requests
import subprocess
from tqdm import tqdm
from collections import OrderedDict

#import sys
#from six.moves import urllib
#def download(url, path):
#    '''
#    wanted to download attributes file
#    mod from https://github.com/SKTBrain/DiscoGAN/blob/master/datasets/download.py
#    '''
#    filename = url.split('/')[-1]
#    filepath = os.path.join(path, filename)
#    u = urllib.request.urlopen(url)
#    f = open(filepath, 'wb')
#    #filesize = int(u.headers["Content-Length"])
#    #print("Downloading: %s Bytes: %s" % (filename, filesize))
#
#    downloaded = 0
#    block_sz = 8192
#    status_width = 70
#    while True:
#        buf = u.read(block_sz)
#        if not buf:
#            print('')
#            break
#        else:
#            print('', end='\r')
#        downloaded += len(buf)
#        f.write(buf)
#        #status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
#        #          ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
#        #print(status, end='')
#        sys.stdout.flush()
#    f.close()
#    return filepath


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                          unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    base_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(base_path)
    os.remove(filepath)

def download_celeb_a(base_path):
    data_path = os.path.join(base_path, 'celebA')
    images_path = os.path.join(data_path, 'images')
    if os.path.exists(data_path):
        print('[!] Found celeb-A - skip')
        return

    filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(base_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    os.rename(os.path.join(base_path, "img_align_celeba"), images_path)
    os.remove(save_path)

    attribute_url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0'
    filepath = download(attribute_url, dirpath)


def prepare_data_dir(path = './data'):
    if not os.path.exists(path):
        os.mkdir(path)

# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)
        os.symlink(rel_link, link_file)

def add_splits(base_path):
    data_path = os.path.join(base_path, 'celebA')
    images_path = os.path.join(data_path, 'images')
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # these constants based on the standard celebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637

    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, test_dir)

if __name__ == '__main__':
    base_path = './data'
    prepare_data_dir()
    download_celeb_a(base_path)
    add_splits(base_path)
