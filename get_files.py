# https://docs.python.org/3/tutorial/datastructures.html#dictionaries
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory

import os

file = open("meta/classes.txt", "r")
training_file = open("training_files.txt", "w")
test_file = open("test_files.txt", "w")

classifications = []

for line in file:
    classification_string = line.split("\n")[0]
    classifications.append(classification_string)

rootDir = "Images/"

for dirName, subdirList, fileList in os.walk(rootDir):

    s = dirName.split("/");
    classification_string = s[len(s) - 1]
    classification_string = classification_string.split("\n")[0]
    
    class_id = 0

    for i in range(0, len(classifications)):
        if(classifications[i] == classification_string):            
            class_id = i
            break

    filenames_classifications = []
    
    for fname in fileList:
        filenames_classifications.append("%s/%s %s\n" % (dirName, fname, class_id))

    # Use 80% of the data for training
    cutoff = 0.8*float(len(filenames_classifications))

    for i in range(0, len(filenames_classifications)):
        if i < cutoff:
            training_file.write(filenames_classifications[i])    
        else:
            test_file.write(filenames_classifications[i])

