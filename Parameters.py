# PARAMETERS ----------------------------------------------------------------- #
#   choosen operators
unary = ["~", "abs", "sin", "cos", "tan", "exp"]
binary = ['-', '+', '*']

#   Tree parameters
INITIALDEPTH = 6
MAXDEPTH = 9

#   GA global parameters
NBGENERATIONS = 300
NBTREES = 100
MUTERATE = 0.8
ELITERATE = 0.15
RANKSELECTION = True
ELITISM = True


#   GA transformations parameters
Brate = 0.70 # probability to generate binary operator over unary one.
CSTrate = 0.20 # probability to generate a constant value instead of variable.
VARrate = 0.05
CSTboundaries = 5 # Maximum absolute value of generable constant value.
# ---------------------------------------------------------------------------- #
