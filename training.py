import random
import math
import sys
from copy import deepcopy
from Parameters import *


# Nodes and Trees containers
Nodes = {}
Trees = {}
# Nodes and Trees indexes
IdNode = 0
IdTree = 0


# GACLASS -------------------------------------------------------------------- #
class GA:
# Constructor
    def __init__(self, dataset):
        # Trees parameters
        self.nbtrees = NBTREES
        self.t_ids = [IdTree + i for i in range(self.nbtrees)]
        self.t_offsprings = []
        self.t_parents = []
        self.ini_depth = INITIALDEPTH
        for i in range(self.nbtrees):
            generate_tree(self.ini_depth)

        # Associated dataset
        self.dataset = dataset

        # GA settings
        self.mute_rate = MUTERATE
        self.elite_rate = ELITERATE
        self.nb_generations = NBGENERATIONS
        self.rankselection = RANKSELECTION
        self.elitism = ELITISM

# Methods
    def run(self):
        for i in range(self.nb_generations):
            self.selection()
            self.crossover()
            self.mutation()
            self.output(i)
        return Trees[self.t_ids[0]].toRPN()

    # Select parents to create next generation
    def selection(self):
        self.update_fitness()
        sumfit = 0

        # Changes fitness for rank selection if needed.
        if self.rankselection == True:
            for i in range(self.nbtrees):
                Trees[self.t_ids[i]].fitness = self.nbtrees - i

        # Select the elites of the generation, as offsprings, if elitism.
        self.t_offsprings = []
        if self.elitism == True:
            nb_elites = math.ceil(self.nbtrees*self.elite_rate)
            if nb_elites % 2 != 0: nb_elites += 1
            for i in range(nb_elites):
                self.t_offsprings.append(Trees[self.t_ids[i]].copy())
        else:
            nb_elites = 0

        # Select the parents for crossover and mutation, thanks to roulette
        # wheel selection
        self.t_parents = []
        sumfit = sum([Trees[i].fitness for i in self.t_ids])
        for i in range(self.nbtrees - nb_elites):
            G = random.uniform(0, sumfit)
            res = 0
            k = 0
            while (res < G):
                res += Trees[self.t_ids[k]].fitness
                k += 1
            self.t_parents.append(Trees[self.t_ids[k-1]].copy())

        # Deletes the old generation
        for tree_id in self.t_ids:
            Trees[tree_id].del_()
        self.t_ids = []

    # Crossover couples of random trees : choose one node from each and swap
    # one of the children from each node (=) Creation of offsprings.
    def crossover(self):
        newparents = []
        for i in range(len(self.t_parents)//2):
            a = random.choice(self.t_parents)
            b = random.choice(self.t_parents)
            offa = Trees[a].copy()
            offb = Trees[b].copy()
            Trees[offa].crossover(Trees[offb])
            newparents += [offa, offb]

        # Deletes the old parents
        for tree_id in self.t_parents:
            Trees[tree_id].del_()

        self.t_parents = deepcopy(newparents)

    # Mutates every tree : choose a random node and generates a random tree
    # starting from this node.
    def mutation(self):
        for i in self.t_parents:
            if random.randint(0,100)/100 < self.mute_rate:
                Trees[i].subtree_mutation()

        # add the new mutated and crossovered parents to offsprings
        self.t_offsprings += self.t_parents
        # new generation becomes the offsprings
        self.t_ids = deepcopy(self.t_offsprings)

    # Updates the fitness of each tree and sort the ids according to fitness
    def update_fitness(self):
        for tid in self.t_ids:
            Trees[tid].update_fitness(self.dataset)
        self.t_ids.sort(key=lambda x: Trees[x].fitness, reverse = True)

    # Displays Output after each generation.
    def output(self, i):
        print(i,"MSE : ",Trees[self.t_ids[0]].compute_MSE(self.dataset), Trees[self.t_ids[0]].toRPN(), file=sys.stderr)

    # toString display, displays the offsprings, parents and previous trees.
    def toString(self):
        print("t_ids : ")
        for i in self.t_ids:
            print(Trees[i].MSE, Trees[i].toString())
        print("t_parents : ")
        for i in self.t_parents:
            print(Trees[i].MSE, Trees[i].toString())
        print("t_offsprings : ")
        for i in self.t_offsprings:
            print(Trees[i].MSE, Trees[i].toString())
        print()
# ---------------------------------------------------------------------------- #



# TREECLASS ------------------------------------------------------------------ #
class Tree:
# Constructor
    def __init__(self):
        global IdTree
        self.id = IdTree
        Trees[IdTree] = self
        IdTree += 1
        self.MSE = -1

# Destructor / Copy
    def del_(self):
        Nodes[self.root].del_()
        del Trees[self.id]

    def copy(self):
        T = Tree()
        n = Nodes[self.root].copy()
        T.set_root(n)
        T.MSE = self.MSE
        return T.id

# Tree Functions
    def set_root(self, nid):
        self.root = nid

    def regulate(self):
        Nodes[self.root].regulate(MAXDEPTH)

    def toRPN(self): # generate RPN expression from tree
        res = RPN("",self.root)
        return res

    def to_list(self):
        return to_list(self.root, [])

    def to_list_parents(self):
        return to_list_parents(self.root, [])

# Evaluation Functions
    def update_fitness(self, dataset):
        self.MSE = self.compute_MSE(dataset)
        self.fitness = 1/self.MSE

    def evaluate(self, Xi):
        return evaluate_RPN(self.toRPN(), Xi)

    def compute_MSE(self, dataset):
        N = dataset[0]
        M = dataset[1]
        X = dataset[2]
        Y = dataset[3]
        MSE = 0
        try:
            for i in range(M):
                MSE += (self.evaluate(X[i]) - Y[i])**2
        except:
            return float('inf')
        return MSE/M

# Genetic Functions
    def crossover(self, tree2):
        id1 = random.choice(self.to_list_parents())
        id2 = random.choice(tree2.to_list_parents())
        child1 = random.choice(Nodes[id1].children)
        child2 = random.choice(Nodes[id2].children)
        Nodes[id1].children.remove(child1)
        Nodes[id2].children.remove(child2)
        Nodes[id1].add_children(child2)
        Nodes[id2].add_children(child1)

    def subtree_mutation(self):
        l = self.to_list()
        nid = random.choice(l)
        d = Nodes[nid].depth()
        for child in Nodes[nid].children:
            Nodes[child].del_()
        Nodes[nid].generate_children(d - 1)
        self.regulate()

# toString Function
    def toString(self):
        return "root = " + str(self.root) + " : " + self.toRPN()
# ---------------------------------------------------------------------------- #



# NODECLASS ------------------------------------------------------------------ #
class Node:
# Constructor
    def __init__(self, depth):
        global IdNode
        self.nid = IdNode
        self.var_rate = VARrate
        self.cst_bound = CSTboundaries
        self.cst_rate = CSTrate
        self.bin_rate = Brate

        if (depth == 1) | (random.randint(0,100)/100 < self.var_rate):
            self.type = "unary"
            if random.randint(0,100)/100 < self.cst_rate:
                self.data = random.randint(-self.cst_bound, self.cst_bound)
            else:
                self.data = "x" + str(random.randint(1, dataset[0]-1))
        else:
            if random.randint(0,100)/100 < self.bin_rate:
                self.type = "binary"
                self.data = random.choice(binary)
            else:
                self.type = "unary"
                self.data = random.choice(unary)

        self.children = []
        Nodes[IdNode] = self
        IdNode += 1

# Destructor / Copy
    def del_(self):
        for child in self.children:
            Nodes[child].del_()
        del Nodes[self.nid]

    def copy(self):
        global IdNode
        n = deepcopy(self)
        n.nid = IdNode
        IdNode += 1
        ch = []
        for child in n.children:
            ch.append(Nodes[child].copy())
        Nodes[n.nid] = n
        n.children = ch
        return n.nid

# Node Functions
    def generate_children(self, depth):
        self.children = []
        if depth > 0:
            self.leaf = False
            child1 = Node(depth)
            self.add_children(child1.nid)

            if self.type == "binary":
                child2 = Node(depth)
                self.add_children(child2.nid)

            for child in self.children:
                Nodes[child].generate_children(depth-1)
        else:
            self.leaf = True

    def add_children(self, nid):
        self.children.append(nid)

    def regulate(self, depth):
        if depth == 0:
            self.del_()
        elif depth == 1:
            for child in self.children:
                Nodes[child].regulate(0)
            self.children = []
        else:
            for child in self.children:
                Nodes[child].regulate(depth-1)

    def depth(self):
        if self.isleaf() == True:
            return 1
        else:
            return 1 + max([Nodes[child].depth() for child in self.children])

    def isleaf(self):
        if len(self.children) == 0:
            return True
        return False

# toString Function
    def toString(self):
        res = " "
        res += str(self.data)
        res += " "
        if self.type == "unary":
            res += "U "
        else:
            res += "B "
        res += "childr : " + str(self.children) + ""
        return res
# ---------------------------------------------------------------------------- #



# FONCTIONS ------------------------------------------------------------------ #
# Deal with the input file for the regression
def input(filename):
    f = open(filename)
    n = len(f.readline().split(","))
    m = 0
    DataX = []
    DataY = []
    line = f.readline().split(",")
    while line != ['']:
        DataX.append(list(map(float,line[:n-1])))
        DataY.append(float(line[-1]))
        line = f.readline().split(",")
        m += 1
    return n, m, DataX, DataY

# Evaluate a Reverse Polish Notation expression according to a given dataset X
def evaluate_RPN(expression, datasetX):
    try:
        stack = []
        for val in expression.split(' '):
            if val in binary:
                op1 = stack.pop()
                op2 = stack.pop()
                if val=='-': r = op2 - op1
                elif val=='+': r = op2 + op1
                elif val=='*': r = op2 * op1
                elif val=='/': r = op2 / op1
                else: r = op2**op1
                stack.append(r)
            elif val in unary:
                op = stack.pop()
                if val=="~": r = - op
                elif val=="abs": r = abs(op)
                else: r = eval("math."+val+"("+ str(op) +")")
                stack.append(r)
            elif val[0] == "x":
                stack.append(datasetX[int(val[1:])-1])
            else:
                stack.append(float(val))
        return stack.pop()
    except:
        return float('inf')

# Recursive function in order to extract RPN from tree
def RPN(r, nid):
    if Nodes[nid].isleaf() == True:
        return str(Nodes[nid].data)
    if Nodes[nid].type == "binary":
        res = RPN(r, Nodes[nid].children[0]) + " "
        res += RPN(r, Nodes[nid].children[1]) + " "
        res += str(Nodes[nid].data)
        return res
    res = RPN(r, Nodes[nid].children[0]) + " "
    res += str(Nodes[nid].data)
    return res

def to_list(nid, list_=[]):
    list_ += [nid]
    for child in Nodes[nid].children:
        to_list(child, list_)
    return list_

def to_list_parents(nid, list_=[]):
    if Nodes[nid].isleaf() == False:
        list_ += [nid]
        for child in Nodes[nid].children:
            to_list_parents(child, list_)
    return list_

def generate_tree(depth, nid = None):
    global IdTree
    T = Tree()
    if nid == None:
        n = Node(depth)
        nid = n.nid
    T.set_root(n.nid)
    Nodes[n.nid].generate_children(depth-1)
    return nid

def toString_N(Nodes):
    for k, v in Nodes.items():
        print(str(k) + v.toString())

def toString_T(Trees):
    for k, v in Trees.items():
        print(str(k) + " root = " + str(v.root))
# ---------------------------------------------------------------------------- #


csvfilename = sys.argv[1]

dataset = input(csvfilename)

g = GA(dataset)

print(g.run())
