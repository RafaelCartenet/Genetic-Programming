import math
import sys

unary = ["~", "abs", "sin", "cos", "tan", "exp"]
binary = ['-', '+', '*']

def input_(filename):
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

def evaluate_RPN(expression, datasetX):
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

def evaluate(Xi, RPN):
    return evaluate_RPN(RPN, Xi)

def compute_MSE(dataset, RPN):
    N = dataset[0]
    M = dataset[1]
    X = dataset[2]
    Y = dataset[3]
    MSE = 0
    for i in range(M):
        MSE += (evaluate(X[i], RPN) - Y[i])**2
    return MSE/M

RPN = sys.argv[1]
dataset = input_(sys.argv[2])

print(compute_MSE(dataset, RPN))
