from collections import Counter
import random
import torch


def extract_sentences(data_dir):
    sentences, sentence = [], []
    with open(data_dir, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                sentences.append(sentence)
                sentence = []
            elif line[0] != "#":
                token = line.split("\t")
                sentence.append(token)
    return sentences

def data_split(data_dir, shuffle=False, split_ratio=(8,1,1)):
    sentences = extract_sentences(data_dir)[:200]
    if shuffle:
        indeces = list(range(len(sentences)))
        random.shuffle(indeces)
        sentences = [sentences[i] for i in indeces] #shuffle the sentences in the dataset
    """the default ratio among training, development, test set is 8:1:1"""
    try:
        if not isinstance(split_ratio,tuple):
            raise TypeError("Invalid split")
        elif not len(split_ratio) == 3:
            raise Exception("Invalid number of split sets")
        for i in split_ratio:
            if not isinstance(i,int):
                raise TypeError("Invalid split")
    except Exception:
        raise
    else:
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                sentences[i][j][2] = int(sentences[i][j][2])
        a,b,c = split_ratio
        b1, b2 = a/(a+b+c), (a+b)/(a+b+c)
    boundary1, boundary2 = round(len(sentences) * b1), round(len(sentences) * b2)
    return sentences[:boundary1], sentences[boundary1:boundary2], sentences[boundary2:]


def create_embeddings(sentences, n1, n2, n3):
    #data_dir = "en-ud-dev.tab"
    vocab, postag, labels = {"Null", "root"}, {"NULL", "_"}, {"NULL", "_"}
    for sentence in sentences:
        for token in sentence:
            vocab.add(token[0])
            postag.add(token[1])
            labels.add(token[3])

    word_to_ix = {word:i for i, word in enumerate(vocab)}
    pos_to_ix = {pos:i for i, pos in enumerate(postag)}
    label_to_ix = {label:i for i, label in enumerate(labels)}

    word_embeds = torch.nn.Embedding(len(vocab),n1) #the number of dimensions for word embeddings
    pos_embeds = torch.nn.Embedding(len(postag),n2)
    label_embeds = torch.nn.Embedding(len(labels),n3)


    def create_type2vec(t,type_to_ix, embeds): #type, type_to_ix, type_embeds
        type2vec, vec2type = {}, {}
        for w in t:
            form = torch.tensor([type_to_ix[w]])
            vec = embeds(form)
            vec = vec.reshape(vec.size()[1])
            type2vec[w] = vec
            vec2type[vec] = w
        return type2vec, vec2type
    word2vec, vec2word = create_type2vec(list(vocab), word_to_ix, word_embeds)
    pos2vec, vec2pos = create_type2vec(list(postag), pos_to_ix, pos_embeds)
    label2vec, vec2label = create_type2vec(list(labels),label_to_ix, label_embeds)

    return word2vec, pos2vec, label2vec









