from openml.apiconnector import APIConnector
import pandas as pd
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

home_dir = os.path.expanduser("~")
openml_dir = os.path.join(home_dir, "openml")
cache_dir = os.path.join(openml_dir, "cache")
with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
    key = fh.readline().rstrip('\n')
openml = APIConnector(cache_directory=cache_dir, apikey=key)
dataset = openml.download_dataset(10)
dataset = openml.download_dataset(10)



X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
lymph = pd.DataFrame(X, columns=attribute_names)
lymph['class'] = y
print(len(lymph))

n, bins, patches = plt.hist(lymph['class'], facecolor='green')
plt.xlabel('class')
plt.grid(True)
plt.show()

lymph = lymph['class']
