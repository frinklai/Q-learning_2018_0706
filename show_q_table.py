import pandas as pd

file_name = 'q_table.pickle'
print file_name
q = pd.read_pickle(file_name)
print q
