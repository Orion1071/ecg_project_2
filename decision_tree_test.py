from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import ecg_library_v1 as ecg
from sklearn.model_selection import train_test_split

# get ecg data

# build tree

def main():
    X_pcl = ecg.unpickler("/scratch/thurasx/ptb_xl/all_waves_data.pcl")
    y = ecg.unpickler("/scratch/thurasx/ptb_xl/reference_scp_list.pcl")

    # print(y)
    v, counts = np.unique(np.array(y), return_counts=True)
    # print(counts)
    np.set_printoptions(threshold=np.inf)
    
    X = []
    for i in range(len(X_pcl)):
        # print(X_pcl[i].shape)
        tmp = X_pcl[i].flatten()
        # print(tmp.shape)
        X.append(tmp)
    print("-----------")

    
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33, random_state=42)
    
    # tmp = ecg.unpickler("/scratch/thurasx/x1.pcl")
    # tmp = tmp[:]
    plt.figure(figsize=(100,10))
    plt.plot(X_train[0])
    plt.savefig("X_train_bc.png")
    print("-----------")
    
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    classifier.fit(X_train, y_train)
    treeplot = tree.plot_tree(classifier)
    plt.figure(dpi=30)
    plt.savefig("tree.png")


if __name__== "__main__" :
    main()