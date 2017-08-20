import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Generate SP algo comparison graphs')
parser.add_argument('first', help='first algo name')
parser.add_argument('second', help='second algo name')

args = parser.parse_args()


print(args.first)
print(args.second)

print ("Comparing " + args.first + " and " + args.second)

reports_root = "../output/" 
path_1 = reports_root + args.first + "_average.csv"
path_2 = reports_root + args.second + "_average.csv"

first_array = []
second_array = []

with open(path_1, 'rb') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in csv_data:
        try:
            first_array.append ([float(i) for i in row])
        except: 
            pass

with open(path_2, 'rb') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in csv_data:
        try:
            second_array.append ([float(i) for i in row])
        except: 
            pass

first_array = np.array(first_array).astype(np.float)
second_array = np.array(second_array).astype(np.float)


# for row in spamreader:
#     print ', '.join(row)


# K   Rec 1 - UE  EV

def plot_value (first_arr, second_arr, index, title, y_label, savepng):
    global args

    f = plt.figure(index)
    plt.plot( first_arr [:,0], first_arr [:,index], 'o-', label=args.first)
    plt.plot(second_arr [:,0], second_arr [:,index], '-x', label=args.second) 


    plt.xlabel('Number of SPs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if savepng:
        plt.savefig("test.png")

    plt.legend()





plot_value (first_array, second_array, 1, "Boundry Recall", "REC", False)
plot_value (first_array, second_array, 2, "Undersegmentation Error", "1 - UE", False)
plot_value (first_array, second_array, 3, "Explained Variation", "EV", False)

plt.show()