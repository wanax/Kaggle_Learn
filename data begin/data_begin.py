#######################

# Machine Learning Test 2018/09/15
# Miles Ma
#
########################

import pandas as pd

data = pd.read_csv('pokemon.csv')

temp = data["HP"]
temp2 = data[["HP"]]

data = data.set_index(["Type 1","Type 2"]) 














































