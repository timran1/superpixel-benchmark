import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Generate SP algo comparison graphs')
parser.add_argument('algo_tuple_list', help='Comma separated algo_tuples list')

args = parser.parse_args()


#print(args.algo_tuple_list)
algo_tuple_list = [x.strip() for x in args.algo_tuple_list.split(',')]

print ("Comparing " + str (algo_tuple_list))

reports_root = "../output/" 
arrays_list = []

for algo_tuple in algo_tuple_list:

    path = reports_root + algo_tuple + "_average.csv"
    array = []

    with open(path, 'rb') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in csv_data:
            try:
                array.append ([float(i) for i in row])
            except: 
                pass

    array = np.array(array).astype(np.float)
    arrays_list.append (array)

#print (arrays_list)

# K   Rec 1 - UE  EV 

def plot_value (arrays, index, title, y_label, savepng):
    global algo_tuple_list

    f = plt.figure(index)
    ptr=0
    for arr in arrays:

        plt.plot(arr [:,0], arr [:,index], 'x-', label=algo_tuple_list[ptr])

        ptr += 1

    plt.xlabel('Number of SPs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if savepng:
        plt.savefig("test.png")

    plt.legend()


plot_value (arrays_list, 1, "Boundry Recall", "REC", False)
plot_value (arrays_list, 2, "Undersegmentation Error", "1 - UE", False)
plot_value (arrays_list, 3, "Explained Variation", "EV", False)

plt.show()