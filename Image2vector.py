import coco
from cache import cache


import os, time, re, math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from spacy.lang.en import English

nlp = English()

stopword = 'stopwords_en.txt'
dir_path = 'data\coco'


def set_path_stopword(newpath):
    global stopword
    stopword = os.path.join(newpath, stopword)

def set_dir_path(newpath):
    global dir_path
    coco.set_data_dir(newpath)
    dir_path = newpath


def remove_stopwords(words, stopwords):
    cleaned_text = [w.lower() for w in words if w not in stopwords]
    return cleaned_text


def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """

    # Load the image using PIL.
    img = Image.open(path)

    # Resize image if desired.
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array.
    img = np.array(img)

    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


def show_image(idx, filenames_val, captions_val, cap=True):


    dir = coco.val_dir
    filename = filenames_val[idx]
    captions = captions_val[idx]

    # Path for the image-file.
    path = os.path.join(dir, filename)

    # Print the captions for this image.
    if cap:
        for caption in captions:
            print(caption)

    # Load the image and plot it.
    img = load_image(path)
    plt.imshow(img)
    plt.show()


def lemma_spacy(str_input):
    doc = nlp(str_input.lower())
    lemma =' '.join(i.lemma_  for i in doc)
    return re.sub( "[^\w]", " ",lemma).split()

def _Image2data(captions, stopwords):
    data = dict()
    voc = []
    for caption in captions:
        #listCap = remove_stopwords(re.sub( "[^\w]", " ",caption).split(),stopwords)
        listCap = remove_stopwords(lemma_spacy(caption),stopwords)

        voc.extend(listCap)
        for word in listCap:
            t = data.get(word)
            if t == None:
                data[word] = 1
            else: data[word] = t+1
    return data, voc


def _load_vector(filenames, captions):

    # load filename and caption from dir val

    # file stopword_en
    f = open(stopword, 'r')
    stopwords = [line.strip() for line in f.readlines()]
    f.close()

    # collect data, vocabulary of each file
    data = dict()
    voc = []
    number_files = len(filenames)
    for i in range(number_files):
        data[i], temp = _Image2data(captions[i], stopwords)
        voc.extend(temp)

    voc = sorted(list(set(voc)))
    number_voc = len(voc)

    BoW = np.zeros((number_voc, number_files), dtype=np.int)
    invert = [[] for _ in range(number_voc)]

    for i in range(number_voc):
        for j in range(number_files):
            freq = data[j].get(voc[i],0)
            if freq != 0:
                BoW[i][j] = freq
                invert[i].append(j)

    Vector = [(BoW[i], voc[i], invert[i]) for i in range(number_voc)]

    f=open('BOW.txt','w')
    cont =''
    for i in BoW:
        cont +=  ' '.join(map(str,i)) + '\n'
    f.write(cont)
    f.close()

    f=open('voc.txt','w')
    cont = '\n'.join(map(str,voc))
    f.write(cont)
    f.close()

    f=open('invert.txt','w')
    cont = '\n'.join(map(str,invert))
    f.write(cont)
    f.close()

    zBoW, zvoc, zinvert = zip(*Vector)
    return zBoW, zvoc, zinvert

def load_vector(filenames, captions):
    cache_filename = "data_vector.pkl"
    cache_path = os.path.join(dir_path, cache_filename)

    vectors = cache(cache_path=cache_path,
                    fn=_load_vector,
                    filenames= filenames,
                    captions= captions)
    return vectors

def _tf_idf(filenames,captions):

    bow, voc, invert = load_vector(filenames, captions)
    number_file = len(filenames)
    number_voc = len(voc)

    tf = np.zeros((number_voc, number_file))
    idf = np.zeros(number_voc)

    for i in range(number_voc):
        idf[i] = round(1+ math.log(number_file/len(invert[i])), 3)
        for j in range(number_file):
            if bow[i][j] != 0:
                tf[i][j] = round(1 + math.log(bow[i][j]), 3)

    records = [(np.dot(tf[i],idf[i]), idf[i])
            for i in range(len(idf))]

    weight, _idf = zip(*records)

    return weight, _idf, voc, invert


def load_weight(filenames, captions):
    cache_filename = 'weight.pkl'
    cache_path = os.path.join(dir_path, cache_filename)

    weight = cache(cache_path= cache_path,
                   fn=_tf_idf,
                   filenames=filenames,
                   captions=captions)
    return weight