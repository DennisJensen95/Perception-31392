import torch
import pandas as pd
from lib.pretrainedNeuralNets import getCNNFeatureExtractVGG19
from lib.getDataGoogle import classes_encoder
import numpy as np
from lib.trainSVMHelper import process_image, extract_features_with_vgg19_cnn
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.decomposition import PCA
import pickle

def main(debug):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    featureExtractor = getCNNFeatureExtractVGG19().to(device)

    dataframe_train = pd.read_csv('self_made_train.csv')
    dataframe_test = pd.read_csv('self_made_test.csv')

    load = True
    do_PCA = True
    path_train = './Results/Saved_SVM_Models/all_train'
    path_test = './Results/Saved_SVM_Models/all_test'
    if not load:
        # Extract training data
        X_train, y_train = extract_features_with_vgg19_cnn(dataframe_train, featureExtractor, device,
                                                           save=True, path=path_train, debug=debug)

        # Extract test data
        X_test, y_test = extract_features_with_vgg19_cnn(dataframe_test, featureExtractor, device,
                                                         save=True, path=path_test, debug=debug)
    else:
        X_train = np.loadtxt(path_train + '_X', delimiter=',')
        y_train = np.loadtxt(path_train + '_y', delimiter=',')

        X_test = np.loadtxt(path_test + '_X', delimiter=',')
        y_test = np.loadtxt(path_test + '_y', delimiter=',')

    if PCA:
        n_components = 30
        t0 = time.time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
        print("done in %0.3fs" % (time.time() - t0))

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    param_grid = {"C": [0.1], "gamma": [0.1]}
    rf = SVC()
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1, verbose=1)
    gs = gs.fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)

    clf = SVC(C=gs.best_params_['C'], kernel='rbf', gamma=gs.best_params_['gamma'])
    clf = clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

    filename = './Results/Saved_SVM_Models/PCA_final_open_image.sav'
    filename_pca = './Results/Saved_SVM_Models/PCA_transform.sav'
    pickle.dump(clf, open(filename, 'wb'))
    pickle.dump(pca, open(filename_pca, 'wb'))

if __name__ == '__main__':
    main(True)