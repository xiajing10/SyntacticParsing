from sys import stdin, stderr
from json import loads, dumps
import numpy as np

def cnf(tree):
    if isinstance(tree,str) or is_cnf(tree):
        pass
    else:
        while len(tree) == 2 and not isinstance(tree[1], str):
            tree[0] = tree[0] + '+' + tree[1][0]
            tree[1:] = tree[1][1:]

        if len(tree) > 3:
            tree[2] = [tree[0]+ '|' + tree[1][0]] + tree[2:]
            while len(tree) > 3:
                tree.pop()
        
        for i in range(1, len(tree)):
            subtree = tree[i]
            cnf(subtree)     
                  

    # pass         

    # CODE REMOVED

    # MASTERS: INSERT YOUR CODE HERE!

    # BACHELORS: CODE ONLY NEEDED FOR ONE OF THE POSSIBLE VG TASKS

def is_cnf(tree):
    n = len(tree)
    if n == 2:
        return isinstance(tree[1], str)
    elif n == 3:
        return is_cnf(tree[1]) and is_cnf(tree[2])
    else:
        return False

def words(tree):
    if isinstance(tree, str):
        return [tree]
    else:
        ws = []
        for t in tree[1:]:
            ws = ws + words(t)
        return ws

if __name__ == "__main__":

    for line in stdin:
        tree = loads(line)
        sentence = words(tree)
        input = str(dumps(tree))
        cnf(tree)
        if is_cnf(tree) and words(tree) == sentence:
            print(dumps(tree))
        else:
            print("Something went wrong!", file=stderr)
            print("Sentence: " + " ".join(sentence), file=stderr)
            print("Input: " + input, file=stderr)
            print("Output: " + str(dumps(tree)), file=stderr)
            exit()


