"""
The function converts labels from mutiple labels to one label
Eg: 0: 1 penny, 1 : 1 dollar, 2: 10 dollars can all be converted to a single label: money
just list the labels to be merged in l1 list
"""
import os

folder = "data/cash_denominations"

train = "/train/labels/"
train_new = "/train_new/labels/"
test = "/test/labels/"
test_new = "/test_new/labels/"
valid = "/valid/labels/"
valid_new = "/valid_new/labels/"
exp = "/experiment/"
exp_dup = "/exp_dupl/"
l1 = [0, 1, 2, 3, 4, 5, 6]
# l2 = [4,5,6,7,8,9]

"""
os.makedirs(folder+test_new)
os.makedirs(folder+train_new)
os.makedirs(folder+valid_new)
"""


def convert(dir, dir_dup):
    files = os.listdir(dir)
    for file in files:
        print("file", str(file))
        fo = open(str(dir + file), "r")
        fn = open(str(dir_dup + file), "w")
        for line in fo:
            if line[0] != "\n":
                val = int(line[0])
            if val in l1:
                fn.write(line.replace(line[0], str(0), 1))
            # elif val in l2:
            # fn.write(line.replace(line[0], str(1),1))
        fo.close()
        fn.close()


convert(folder + train, folder + train_new)
