import coco
import Image2vector as i2v


import numpy as np
import re, time


def combine(vector, invert):
	temp = []
	for index in range(len(vector)):
		if vector[index] != 0: 
			temp.extend(invert[index])
	return list(set(temp))


def negate(arr, number_file):
    temp=[i
          for i in range(number_file)
          if i not in arr]
    return list(temp)


def special(str):
    begin = re.search(r"\"\b", str)
    end = re.search(r"\"\b", str[:][:][::-1])
    if  begin != None: begin = begin.end()
    if end != None:  end = len(str) - end.start()-1
    return (begin, end)


def standardized(Text):
    listText = re.sub("[^\w]", " ", Text.lower()).split()
    return listText


def vector_intersect(vector, invert):
    arr = np.arange(1,len(vector),1)
    for index in range(len(vector)):
        if vector[index] != 0:
            arr = np.intersect1d(np.array(arr), np.array(invert[index]))
    return arr


def Text2vector(Text, voc):
    #listText = standardized(Text)
    listText = i2v.lemma_spacy(Text)
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


def Norm2(Query, BoW, voc, invert):
    vector = Text2vector(Query,voc)
    BoW = np.array(BoW).T
    spe = special(Query)
    plus = []
    substring = []
    vector = Text2vector(Query, voc)
    if spe[0] != None:
        plus = vector_intersect(vector, invert)

        #Query = Query[:spe[0]] + Query[spe[1]:]

    plus = vector_intersect(vector, invert)

    m_content = standardized(Query)
    weight = []
#    list_nameInverted = []
#    for word in m_content:
#        if word in voc:
#            i = voc.index(word)
#            w = 1
#            if word in substring:
#                w = 0.5
#            for name in invert[i]:
#                if name in list_nameInverted:
#                    indw = list_nameInverted.index(name)
#                else:
#                    print(name)
#                    list_nameInverted.append(name)
#                    weight.append(w)

    list_nameInverted = list(set(plus))

    distance = dict()
    for i in range(len(list_nameInverted)):
        distance[list_nameInverted[i]] = np.linalg.norm(BoW[list_nameInverted[i]] - vector)
    dist = sorted(distance, key=distance.get)
    return  dist


def distance_tf_idf(Query, weight, idf, voc):
    tf_query = Text2vector(Query, voc) * idf
    vector = np.array(weight).T
    dist = dict()
    for index in range(len(vector)):
        dist[index] =  np.linalg.norm(vector[index] - tf_query)
    dist = sorted(dist,key=dist.get)
    return dist


def distance(Query, bow, voc):
    bow = np.array(bow).T
    vt = Text2vector(Query, voc)

    Distance = dict()
    for i in range(len(bow)):
        Distance[i] = np.linalg.norm(bow[i] - vt)

    dist = sorted(Distance,key=Distance.get)
    return dist


def consine(vectorQuery, weightT, voc, idf, norm2, ListIDdoc):
    tic=time.time()

    norm2_query = np.linalg.norm(vectorQuery)
    if norm2_query == 0:return []
    consine = dict()
    for index in range(len(ListIDdoc)):
        consine[ListIDdoc[index]] = np.dot(weightT[ListIDdoc[index]],vectorQuery)/(norm2[ListIDdoc[index]]* norm2_query)
    consine = sorted(consine, key=consine.get, reverse=True)
    # print('consine ', time.time() - tic)
    return consine


def decode_query(Query, weightT, voc, idf, norm2, inverted):
    spe = special(Query)
    vector_query = Text2vector(Query, voc)*idf
    plus = combine(vector_query, inverted)
    if spe[0] != None and spe[1] != None:
        vector = Text2vector(Query[spe[0]: spe[1]], voc)
        plus = vector_intersect(vector, inverted)
        if Query[spe[0]-2] == '-':
            plus = negate(plus,len(weightT))
    return consine(vector_query, weightT, voc, idf, norm2, plus)


tic = time.time()
# path file stopwords_en.txt
stopword_path = 'D:\source\PythonProject\IR'
# path dir data
dir_path = 'D:\data\COCO'

i2v.set_path_stopword(stopword_path)
i2v.set_dir_path(dir_path)


_, filenames_val, captions_val = coco.load_records(train=False)
#BoW, voc, invert = i2v.load_vector(filenames_val, captions_val)
#BoW = np.array(BoW).T


weight, idf, voc, invert = i2v.load_weight(filenames_val,captions_val)


weightT = np.array(weight).T
norm2 = [np.linalg.norm(i)
         for i in weightT]

print("time to load: ", time.time() - tic)

while True:
    Query = input('Nhap cau hoi: ')
    if Query == '--exit':break
    tic= time.time()
    #idx =#distance(Query,weight, idf, voc, filenames_val)
    #idx = Norm2(Query, BoW, voc,invert)
    #idx = distance(Query,BoW, voc)
    idx = decode_query(Query, weightT, voc, idf, norm2, invert)
    print('Có ',len(idx)," kết quả (", round(time.time() - tic,2), 's)',sep='')
    for i in range(len(idx)):
        if i > 3: break
        i2v.show_image(idx[i],filenames_val, captions_val, False)
