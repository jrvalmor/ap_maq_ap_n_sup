import os

path = r'H:\\Trabalho_Final\\aprender2\\ORL_2'

def listDir(dir):
    fileNames = os.listdir(dir)
    for fileName in fileNames:
        print("Nome_arquivo: " + fileName)

if __name__ == '__main__':
    listDir(path)
