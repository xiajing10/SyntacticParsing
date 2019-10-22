SH = 0; RE = 1; RA = 2; LA = 3
labels = ["det", "nsubj", "case", "nmod", "root"]

def attach_orphans(arcs, n):
    attached = []
    for (h, d, l) in arcs:
        attached.append(d)
    for i in range(1, n):
        if not i in attached:
            arcs.append((0, i, "root"))

def print_tree(root, arcs, words, indent):
    if root == 0:
        print(" ".join(words[1:]))
    children = [(root, i, l) for i in range(len(words)) for l in labels if (root, i, l) in arcs]
    for (h, d, l) in sorted(children):
        print(indent + l + "(" + words[h] + "_" + str(h) + ", " + words[d] + "_" + str(d) + ")")
        print_tree(d, arcs, words, indent + "  ")

def transition(trans, stack, buffer, arcs):
    if isinstance(trans, int):
        if trans == SH:
            stack.insert(0, buffer.pop(0))
        elif trans == RE:
            stack.pop(0)
    elif isinstance(trans, tuple):
        if trans[0] == LA:
            arcs.append((buffer[0],stack[0],trans[1]))
            stack.pop(0)
        elif trans[0] == RA:
            arcs.append((stack[0],buffer[0],trans[1]))
            stack.insert(0, buffer.pop(0))
    # add code for missing transitions: RE, (RA, label), (LA, label)

def parse():
    words = "root the cat is on the mat today".split()
    stack = [0]
    buffer = [x for x in range(1, len(words))]
    arcs = []
    for trans in [SH, (LA, "det"), SH, (LA, "nsubj"), SH, SH, SH, (LA, "det"), (LA, "case"), (RA, "nmod"), RE, (RA, "nmod")]:
        transition(trans, stack, buffer, arcs)
    attach_orphans(arcs, len(words))
    print_tree(0, arcs, words, "")

if __name__ == "__main__":
    parse()
