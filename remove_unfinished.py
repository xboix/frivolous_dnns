import os
import sys

directory = str(sys.argv[1])

if directory[-1] != '/':
    directory += '/'

filenames = []
IDs = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".out"):
        with open(directory+filename, 'r') as f:
            lastline = f.readlines()[-1]
            if lastline != ':)\n':
                filenames.append(filename)
                start = filename.find('_')
                end = filename.find('.')
                ID = filename[start+1: end]
                IDs.append(int(ID))
                os.remove(filename)

print(filenames)
print(sorted(IDs))
