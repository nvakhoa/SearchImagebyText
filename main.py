import coco
import Image2vector as i2v


import numpy as np
import re, time

def Text2vector(Text, voc):
    listText =  re.sub("[^\w]"," ",Text.lower()).split()

    n_voc = len(voc)
    bow = np.zeros(n_voc, dtype=np.int)
    for word in range (n_voc):
        if voc[word] in listText:
            count = 0
            while voc[word] in listText:
                count += 1
                listText.remove(voc[word])
            bow[word] = count
    return bow


def Norm2(Query, BoW, voc):
    Query = Text2vector(Query,voc)

    distance = dict()
    for i in range(len(BoW)):
        distance[i] = np.linalg.norm(BoW[i] - Query)

    dist = sorted(distance, key=distance.get)
    return dist

tic = time.time()
# path file stopwords_en.txt
stopword_path = 'D:\source\PythonProject\IR'
# path dir data
dir_path = 'D:\data\COCO'

i2v.set_path_stopword(stopword_path)
i2v.set_dir_path(dir_path)


_, filenames_val, captions_val = coco.load_records(train=False)
BoW, voc, invert = i2v.load_vector(filenames_val, captions_val)

BoW = np.array(BoW).T
print("time to load: ", time.time() - tic)

tic= time.time()
Query = 'bear'

idx = Norm2(Query, BoW, voc)
print("time search : ", time.time() - tic)


for i in range(len(idx)):
    if i > 5: break
    i2v.show_image(i,filenames_val, captions_val)
