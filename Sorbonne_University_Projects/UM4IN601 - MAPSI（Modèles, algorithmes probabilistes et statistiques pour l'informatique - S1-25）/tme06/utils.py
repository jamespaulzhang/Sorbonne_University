import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename:str):
    with open(filename, 'rb') as f:
        data = pkl.load(f, encoding='latin1') 
    X = data.get('letters') # récupération des données sur les lettres
    Y = data.get('labels') # récupération des étiquettes associées 
    return X,Y

def draw_char(let,titre,save=False):
    # affichage d'une lettre
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure(figsize=(2,2))
    plt.title(titre)
    plt.plot(coord[:,0],coord[:,1])
    if save:
      plt.savefig("exlettre.png")
    return

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(X, y, pc):
    np.random.seed(1000)
    indTrainC = {}
    indTestC = {}
    
    indTrain = []
    indTest = []
    yTrain = []
    yTest = []
    
    for cl in np.unique(y): # pour toutes les classes
        ind, = np.where(y == cl)
        n = len(ind)        
        indTrainC[cl]=ind[np.random.permutation(n)][:int(np.floor(pc * n))]
        indTestC[cl]=np.setdiff1d(ind, indTrainC[cl])
                
        for i in range(len(indTrainC[cl])):
            yTrain.append(cl)
            indTrain.append(indTrainC[cl][i])
        
        for i in range(len(indTestC[cl])):
            yTest.append(cl)
            indTest.append(indTestC[cl][i])
    
    X_train = [ X[indTrain[i]] for i in range(len(indTrain))]
    X_test = [X[indTest[i]] for i in range(len(indTest))]

    
    return X_train, X_test, np.array(yTrain),np.array(yTest)
