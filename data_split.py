import numpy as np

data = np.loadtxt("Home1/train_data_7day.csv", delimiter=',')
np.savetxt("Home1/train_data_7day_threshold.csv", data[:60000], delimiter=',')
np.savetxt("Home1/train_data_7day_test.csv", data[60000:], delimiter=',')