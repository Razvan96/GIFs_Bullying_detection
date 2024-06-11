import os
path = 'C:\\Work\\ML\\Video_Classification\\dataset\\bullying'
#path = '/Users/myName/Desktop/directory'
files = os.listdir(path)


for index, file in enumerate(files):
    if index < 10:
        os.rename(os.path.join(path, file), os.path.join(path, ''.join(["bullying_00", str(index), '.gif'])))
    else:
        os.rename(os.path.join(path, file), os.path.join(path, ''.join(["bullying_0", str(index), '.gif'])))