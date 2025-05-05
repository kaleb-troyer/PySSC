
import numpy as np


def funct(To, Ti): 
    return np.cos((np.log(To) - (Ti * 0.28989363)) / To) - 0.24336664

print(funct(900, 590))

