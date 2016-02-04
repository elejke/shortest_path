# import pandas as pd
# import numpy as np
from utils import layers, data_prepare

chip_1, chip_2, connections = data_prepare()

new_points, layers = layers(chip_1, chip_2, connections)

print layers
