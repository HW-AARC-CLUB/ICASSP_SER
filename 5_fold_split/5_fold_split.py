import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

root_path = './csv_file/'
csv_name = 'IEMOCAP_All_with_Path.csv'
