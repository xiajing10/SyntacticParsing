

import torch
import torch.nn as nn
from torch.autograd import Variable
from util import  data_split, create_embeddings
from time import time
from copy import deepcopy
from collections import defaultdict



class MLPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MLPerceptron, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 4)

    def forward(self, inputs):
        w = Variable(inputs[0], requires_grad = True)
        t = Variable(inputs[1], requires_grad = True)
        l = Variable(inputs[2], requires_grad = True)
        inputs = torch.cat([w,t,l],dim = 0)
        temp_vec = self.linear1(inputs)
        temp_vec = torch.tanh(temp_vec)
        results = self.linear2(temp_vec)

        return torch.softmax(results, -1)



class Configuration:
    to_target = {}
    out = torch.zeros([4],dtype=torch.float)
    for i in range(4):
        out_copy = deepcopy(out)
        out_copy[i] = 1.0
        to_target[i] = out_copy

    def __init__(self, sentence):
        self.sentence = sentence
        if sentence[0] != ["root", "_", "0", "_"]:
            self.sentence.insert(0, ["root", "_", "0", "_"])
        self.stack = [0]
        self.arcs = []
        self.buffer = list((range(1,len(sentence))))
        if len(sentence[0]) > 2:
            self.heads = [token[2] for token in sentence]
            self.labels = [token[3] for token in sentence]
        else:
            self.heads, self.labels = None, None
        self.trans_sequence = []
        self.features = []

        def reach_indeces(heads):
            index_dict = defaultdict(list)
            for index, value in enumerate(heads):
                if value in index_dict:
                    index_dict[value].append(index)
                else:
                    index_dict[value] = [index]
            return index_dict

        self.index_dict = reach_indeces(self.heads)

    def oracle(self, stack, buffer, heads, labels):
        SH, SW, RA, LA = 0, 1, 2, 3
        if len(stack) == 1:
            return SH
        if len(buffer) == 0 and len(stack) == 2:
            return RA, labels[stack[0]]
        j, i = stack[0], stack[1]
        if heads[i] == j and j != 0 and i not in heads:
            heads[i] = None
            return LA, labels[i]
        elif heads[j] == i and i != 0 and j not in heads:
            heads[j] = None
            return RA, labels[j]
        else:
            if j > i > 0 and heads[j] in stack[2:]:
                return SW
            return SH

    def extract_features(self):
        features = []
        words, tags, labels = [], [], []

        if len(self.stack) == 0:
            for i in range(15):
                words.append("Null")
                tags.append("NULL")
            for i in range(12):
                labels.append("NULL")

        elif len(self.stack) == 1:
            words.append(self.sentence[self.stack[0]][0])
            tags.append(self.sentence[self.stack[0]][1])
            for i in range(2):
                words.append("Null")
                tags.append("NULL")

        elif len(self.stack) == 2:
            for i in self.stack:
                words.append(self.sentence[i][0])
                tags.append(self.sentence[i][1])
            words.append("Null")
            tags.append("NULL")
        else:
            for i in self.stack[:3]:
                words.append(self.sentence[i][0])
                tags.append(self.sentence[i][1])

        if len(self.buffer) == 0:
            for i in range(3):
                words.append("Null")
                tags.append("NULL")
        elif len(self.buffer) == 1:
            words.append(self.sentence[self.buffer[0]][0])
            tags.append(self.sentence[self.buffer[0]][1])
            for i in range(2):
                words.append("Null")
                tags.append("NULL")
        elif len(self.buffer) == 2:
            for i in self.buffer:
                words.append(self.sentence[i][0])
                tags.append(self.sentence[i][1])
            words.append("Null")
            tags.append("NULL")
        else:
            for i in self.buffer[:3]:
                words.append(self.sentence[i][0])
                tags.append(self.sentence[i][1])

        if len(self.stack) == 1:
            children = self.index_dict[self.stack[0]]
            if children:
                left, right = [], []
                for child in children:
                    if child < self.stack[0]:
                        left.append(child)
                    elif child > self.stack[0]:
                        right.append(child)
                if left:
                    if len(left) > 1:
                        for i in left[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])
                    else:
                        words.append(self.sentence[left[0]][0])
                        tags.append(self.sentence[left[0]][1])
                        labels.append(self.sentence[left[0][3]])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    leftchildren = self.index_dict[left[0]]
                    if leftchildren:
                        leftc = []
                        for child in leftchildren:
                            if child < left[0]:
                                leftc.append(child)

                        if leftc:
                            words.append(self.sentences[leftc[0]][0])
                            tags.append(self.sentence[leftc[0]][1])
                            labels.append(self.sentence[leftc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")

                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")


                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                if right:
                    if len(right) > 1:
                        right.reverse()
                        for i in right[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])
                    else:
                        words.append(self.sentence[right[0]][0])
                        tags.append(self.sentence[right[0]][1])
                        labels.append(self.sentence[right[0]][3])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    rightchildren = self.index_dict[right[0]]
                    if rightchildren:
                        rightc = []
                        for child in rightchildren:
                            if child > right[0]:
                                rightc.append(child)

                        if rightc:
                            rightc.reverse()
                            words.append(self.sentence[rightc[0]][0])
                            tags.append(self.sentence[rightc[0]][1])
                            labels.append(self.sentence[rightc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")
                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
            else:
                for i in range(6):
                    words.append("Null")
                    tags.append("NULL")
                    labels.append("NULL")

            for i in range(6):
                words.append("Null")
                tags.append("NULL")
                labels.append("NULL")

        elif len(self.stack) > 1:
            children = self.index_dict[self.stack[0]]
            if children:
                left, right = [], []
                for child in children:
                    if child < self.stack[0]:
                        left.append(child)
                    elif child > self.stack[0]:
                        right.append(child)
                if left:
                    if len(left) > 1:
                        for i in left[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])
                    else:
                        words.append(self.sentence[left[0]][0])
                        tags.append(self.sentence[left[0]][1])
                        labels.append(self.sentence[left[0]][3])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    leftchildren = self.index_dict[left[0]]
                    if leftchildren:
                        leftc = []
                        for child in leftchildren:
                            if child < left[0]:
                                leftc.append(child)

                        if leftc:
                            words.append(self.sentence[leftc[0]][0])
                            tags.append(self.sentence[leftc[0]][1])
                            labels.append(self.sentence[leftc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")
                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                if right:
                    if len(right) > 1:
                        right.reverse()
                        for i in right[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])
                    else:
                        words.append(self.sentence[right[0]][0])
                        tags.append(self.sentence[right[0]][1])
                        labels.append(self.sentence[right[0]][3])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    rightchildren = self.index_dict[right[0]]
                    if rightchildren:
                        rightc = []
                        for child in rightchildren:
                            if child > right[0]:
                                rightc.append(child)

                        if rightc:
                            rightc.reverse()
                            words.append(self.sentence[rightc[0]][0])
                            tags.append(self.sentence[rightc[0]][1])
                            labels.append(self.sentence[rightc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")
                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
            else:
                for i in range(6):
                    words.append("Null")
                    tags.append("NULL")
                    labels.append("NULL")
            children = self.index_dict[self.stack[1]]
            if children:
                left, right = [], []
                for child in children:
                    if child < self.stack[1]:
                        left.append(child)
                    elif child > self.stack[1]:
                        right.append(child)
                if left:
                    if len(left) > 1:
                        for i in left[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])

                    else:
                        words.append(self.sentence[left[0]][0])
                        tags.append(self.sentence[left[0]][1])
                        labels.append(self.sentence[left[0]][3])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    leftchildren = self.index_dict[left[0]]
                    if leftchildren:
                        leftc = []
                        for child in leftchildren:
                            if child < left[0]:
                                leftc.append(child)

                        if leftc:
                            words.append(self.sentence[leftc[0]][0])
                            tags.append(self.sentence[leftc[0]][1])
                            labels.append(self.sentence[leftc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")
                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                if right:
                    if len(right) > 1:
                        right.reverse()
                        for i in right[:2]:
                            words.append(self.sentence[i][0])
                            tags.append(self.sentence[i][1])
                            labels.append(self.sentence[i][3])
                    else:
                        words.append(self.sentence[right[0]][0])
                        tags.append(self.sentence[right[0]][1])
                        labels.append(self.sentence[right[0]][3])
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")

                    rightchildren = self.index_dict[right[0]]
                    if rightchildren:
                        rightc = []
                        for child in rightchildren:
                            if child > right[0]:
                                rightc.append(child)

                        if rightc:
                            rightc.reverse()
                            words.append(self.sentence[rightc[0]][0])
                            tags.append(self.sentence[rightc[0]][1])
                            labels.append(self.sentence[rightc[0]][3])
                        else:
                            words.append("Null")
                            tags.append("NULL")
                            labels.append("NULL")
                    else:
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
                else:
                    for i in range(3):
                        words.append("Null")
                        tags.append("NULL")
                        labels.append("NULL")
            else:
                for i in range(6):
                    words.append("Null")
                    tags.append("NULL")
                    labels.append("NULL")

        return words+tags+labels


    def transition(self, trans):
        SH, SW, RA, LA = 0, 1, 2, 3
        if isinstance(trans, int):
            if trans == SH:
                self.stack.insert(0, self.buffer.pop(0))
            elif trans == SW:
                self.buffer.insert(0, self.stack.pop(1))
        elif isinstance(trans, tuple):
            if trans[0] == RA:
                self.arcs.append((self.stack[1], self.stack[0], trans[1]))
                self.stack.pop(0)
            elif trans[0] == LA:
                self.arcs.append((self.stack[0], self.stack[1], trans[1]))
                self.stack.pop(1)

    def attach_orphans(self, arcs, n):
        attached = []
        for (h, d, l) in arcs:
            attached.append(d)
        for i in range(1, n):
            if not i in attached:
                arcs.append((0, i, "root"))

    def parse(self):
        words = [token[0] for token in self.sentence]
        arcs = []
        if self.stack:
            trans = self.oracle(self.stack, self.buffer, self.heads, self.labels)
            if isinstance(trans, int):
                self.trans_sequence.append(self.to_target[trans])
            else:
                self.trans_sequence.append(self.to_target[trans[0]])
            feature = self.extract_features()
            self.features.append(deepcopy(feature))
            self.transition(trans)
            while len(self.stack) > 1:
                trans = self.oracle(self.stack, self.buffer, self.heads, self.labels)

                if isinstance(trans, int):
                    self.trans_sequence.append(self.to_target[trans])
                else:
                    self.trans_sequence.append(self.to_target[trans[0]])
                feature = self.extract_features()
                self.features.append(deepcopy(feature))
                self.transition(trans)
        self.attach_orphans(self.arcs, len(words))



def feature2vec(features):
    global word2vec
    global pos2vec
    global label2vec
    words,pos,labels = features[:18], features[18:36], features[36:]
    words = [word2vec[word] for word in words]
    pos = [pos2vec[p] for p in pos]
    labels = [label2vec[label] for label in labels]

    def feature_dimension_convert(tx):
        outx = []

        for i in tx:
            for j in i:
                outx.append(j)
        return torch.tensor(outx, dtype=torch.float)

    wx = feature_dimension_convert(words)
    px = feature_dimension_convert(pos)
    lx = feature_dimension_convert(labels)

    return [wx,px,lx]


def margin_convert_to_prediction(margins):
    new_margins = []

    for margin in margins:
        prediction = torch.zeros([4], dtype=torch.float)
        for i in range(len(margin)):
            if margin[i] == max(margin):
                prediction[i] = 1.
                break

        new_margins.append(prediction)
    return new_margins


def accuracy(x, y):
    assert (len(x) == len(y))
    total = len(x)
    r = 0
    for i, j in zip(x, y):
        if list(i) == list(j):
            r += 1
    return r / total


def train(train_x, val_x, \
          nfeatures = None, hidden_size=100, \
          loss_function=torch.nn.BCEWithLogitsLoss(), \
          learning_rate=0.0001, \
          weight_decay=0.1, \
          n_epoch=10, verbose=1):

    net = MLPerceptron(nfeatures, hidden_size)  # Here we instantiate our Net

    # 'Criterion'
    criterion = loss_function

    if verbose:
        print('Beginning training.')

    # Here store the parameters in the whole training processing
    # with the best observed model with respect to accuracy
    best_val_accuracy = 0


    epoch_start = time()
    n = 0
    while n < n_epoch:

        train_loss, val_loss = [], []
        train_margins, val_margins = [], []
        train_ys, val_ys = [], []
        for sentence in train_x:
            config = Configuration(sentence)
            config.parse()
            for features, train_y in zip(config.features, config.trans_sequence):

                margins = net.forward(feature2vec(features))
                train_margins.append(margins)
                train_ys.append(train_y)
                optimizer = torch.optim.ASGD(net.parameters(), learning_rate)

                training_loss = criterion(margins, train_y)

                train_loss.append(training_loss.item())
                training_loss.backward()
                optimizer.step()

        # Calculates validation loss
        with torch.no_grad():  # We don't want to backprop this!
            for sentence in val_x:
                config = Configuration(sentence)
                config.parse()
                for features, val_y in zip(config.features, config.trans_sequence):
                    val_margin = net.forward(feature2vec(features))
                    val_margins.append(val_margin)
                    val_ys.append(val_y)
                    validation_loss = criterion(val_margin, val_y)
                    val_loss.append(validation_loss.item())

        epoch_end = time()

        # This part generates statistics
        train_predictions = margin_convert_to_prediction(train_margins)
        train_accuracy = accuracy(train_predictions, train_ys)


        val_predictions = margin_convert_to_prediction(val_margins)
        val_accuracy = accuracy(val_predictions, val_ys)

        training_loss = sum(train_loss)/len(train_loss)
        val_loss = sum(val_loss)/len(val_loss)
        epoch_time = epoch_end - epoch_start

        
        print("Epoch %d, training_acc %f, val_acc, %f, training_loss, %f, val_loss, %f, time, %d" % \
              (n, train_accuracy, val_accuracy, training_loss, val_loss, epoch_time))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(net.state_dict(),"best_model")
            #best_val_params = deepcopy(net.parameters)
            #best_model.parameters = best_val_params

        optimizer.zero_grad()
        n += 1

    if verbose:
        print('Training complete.')

    # output of net
    model = MLPerceptron(nfeatures, hidden_size)
    model.load_state_dict(torch.load("best_model"))
    model.eval
    return model



def main():

    """Hyperparamters for the mlp model"""
    # Regularisation strength
    weight_decay = 0.0001

    # Learning rate
    learning_rate = 0.00001

    #N Number of training epochs
    epochs = 40

    # The size of the hidden layer
    hidden_size = 100

    # Loss function
    loss_function = torch.nn.BCEWithLogitsLoss()

    # Batch size is flexible with respect to the length of transition sequence.

    # Embedding size for words, parts of speech and labels.
    words = 20
    pos = 4
    label = 4

    enable_test_set_scoring = True

    # End of hyperparameters

    data_dir = "en-ud-dev-refined.txt"
    train_data, dev_data, test_data = data_split(data_dir)
    global word2vec
    global pos2vec
    global label2vec
    word2vec, pos2vec, label2vec = create_embeddings(train_data + dev_data + test_data, words, pos, label)
    model = train(train_x=train_data, val_x=dev_data, \
                  nfeatures = (words+pos)*18+ label*12, \
                  hidden_size = hidden_size, \
                  loss_function = loss_function, \
                  learning_rate = learning_rate, \
                  weight_decay = weight_decay, \
                  n_epoch = epochs, verbose = 1)

    if enable_test_set_scoring:
        testing_loss, test_margins, test_ys = [], [], []
        with torch.no_grad():
            for sentence in test_data:
                config = Configuration(sentence)
                config.parse()
                for features, test_y in zip(config.features, config.trans_sequence):
                    test_margin = model.forward(feature2vec(features))
                    test_margins.append(test_margin)
                    test_ys.append(test_y)
                    test_loss = loss_function(test_margin, test_y)
                    testing_loss.append(test_loss.item())

        test_predictions = margin_convert_to_prediction(test_margins)
        test_accuracy = accuracy(test_predictions, test_ys)
        test_loss = sum(testing_loss) / len(testing_loss)

        print("Test set acc %f, test_loss, %f" % \
              (test_accuracy, test_loss))

if __name__ == "__main__":
    main()






