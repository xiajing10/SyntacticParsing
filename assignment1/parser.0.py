"""
CKY algorithm from the "Natural Language Processing" course by Michael Collins
https://class.coursera.org/nlangp-001/class
"""
import sys
from sys import stdin, stderr
from time import time
from json import dumps

from collections import defaultdict
from pprint import pprint

from pcfg import PCFG
from tokenizer import PennTreebankTokenizer

def argmax(lst):
    return max(lst) if lst else (0.0, None)

def backtrace(back, bp):
    # ADD YOUR CODE HERE
    # Extract the tree from the backpointers
    if len(back) == 6:
        (C, C1, C2, min_, mid_, max_) = back
        return [C, backtrace(bp[min_ ][mid_][C1], bp),
                    backtrace(bp[mid_][max_][C2], bp)]
    elif len(back) ==4:
        (C, w, min_, max_) = back
        return [C, w]

def CKY(pcfg, norm_words):
    # ADD YOUR CODE HERE
    # IMPLEMENT CKY

    # NOTE: norm_words is a list of pairs (norm, word), where word is the word 
    #       occurring in the input sentence and norm is either the same word, 
    #       if it is a known word according to the grammar, or the string _RARE_. 
    #       Thus, norm should be used for grammar lookup but word should be used 
    #       in the output tree.

    # Initialize your charts (for scores and backpointers)
    n = len(norm_words)
    pi = defaultdict(float) 
    chart = [[ defaultdict(float)  for _ in range(n+1)] for _ in range(n)]
    bp = [[ defaultdict(tuple) for _ in range(n+1)] for _ in range(n)]
    for i in range(n):
        norm, word = norm_words[i]
        for C in pcfg.N:
            if (C, norm) in pcfg.q1:
                # print(i,i+1,C)
                chart[i][i+1][C] = pcfg.q1[(C, norm)]
                # print(chart[i][i+1][C])
                bp[i][i+1][C] = (C, word, i+1, i)
    # Code for the dynamic programming part, where larger and larger subtrees are built
    for max_ in range(2, n+1):
        for min_ in range(max_-2, -1, -1):
            for C in pcfg.binary_rules.keys():
                best = 0
                for C1,C2 in pcfg.binary_rules[C]:
                    for mid_ in range(min_+1, max_):
                        t1 = chart[min_][mid_][C1]
                        t2 = chart[mid_][max_][C2]
                        # print(t1,t2,min_,mid_,max_)
                        if t1*t2 > 0.0:
                            candidate = t1 * t2 * pcfg.q2[C, C1, C2]
                            if candidate > best:
                                best = candidate
                                backpointer = (C, C1, C2, min_, mid_, max_)
                if best > 0.0:
                    chart[min_][max_][C] = best
                    bp[min_][max_][C]= backpointer

            
    
        
    # Below is one option for retrieving the best trees, assuming we only want trees with the "S" category
    # This is a simplification, since not all sentences are of the category "S"
    # The exact arguments also depends on how you implement your back-pointer chart.
    # Below it is also assumed that it is called "bp"
    # return backtrace(bp[0, n, "S"], bp) 

    # _, top = max([(chart[0][n][X], bp[0][n][X]) for X in pcfg.N])
    return backtrace(bp[0][n]["S"], bp)


class Parser:
    def __init__(self, pcfg):
        self.pcfg = pcfg
        self.tokenizer = PennTreebankTokenizer()
    
    def parse(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        norm_words = []
        for word in words:                # rare words normalization + keep word
            norm_words.append((self.pcfg.norm_word(word), word))
        tree = CKY(self.pcfg, norm_words)
        if tree:
            tree[0] = tree[0].split("|")[0]
        return tree
    
def display_tree(tree):
    pprint(tree)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python3 parser.py GRAMMAR")
        exit()

    start = time()
    grammar_file = sys.argv[1]
    print("Loading grammar from " + grammar_file + " ...", file=stderr)    
    pcfg = PCFG()
    pcfg.load_model(grammar_file)
    parser = Parser(pcfg)

    print("Parsing sentences ...", file=stderr)
    for sentence in stdin:
        tree = parser.parse(sentence)
        print(dumps(tree))
    print("Time: (%.2f)s\n" % (time() - start), file=stderr)
