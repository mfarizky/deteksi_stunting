import matplotlib.pyplot as plt
import numpy as np

class KNN:
  def __init__(self, k):
    self.K=k

  def train(self, X, y):
    self.X_train = X
    self.y_train = y

  def euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

  def _predict(self, x):
    #menghitung jarak titik ke seluruh data training
    distance_point = [self.euclidean_distance(x_train,x) for x_train in self.X_train]
    # mengurutkan berdasarkan jarak tetangga terdekat sebanyak K
    k_neighbors = np.argsort(distance_point)[:self.K]
    # ambil label k_neighbors
    label = [self.y_train[i] for i in k_neighbors]
    # mengembalikan label dengan kemunculan terbanyak
    return self.modusValue(label)

  def predict(self, X):
    self.y_predict = np.array([self._predict(x) for x in X]).reshape(-1,1)
    return self.y_predict
  
  def modusValue(self, arr):
      array = np.zeros(np.max(arr)+1)
      for i in arr:
        array[int(i)]+=1
      max = np.max(array)
      for modus in range(len(array)+1):
        if(max == array[modus]): 
          return modus
  
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        
        # Menyimpan nilai fitur pada decision tree
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Nilai untuk setiap Node leaf/daun
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    def train(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        y_pred = [self._predict(x, self.root) for x in X]
        return y_pred
    
    def _predict(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)
        
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # Menghitung nilai node leaf/daun
        leaf_value = self.leaf_value(Y)
        # Mengembalikan nilai node leaf/daun
        return Node(value=leaf_value)
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def get_best_split(self, dataset, num_features):
        
        # Membuat dictionary untuk menyimpan kumpulan pembagian fitur terbaik
        best_split = {}
        max_info_gain = -float("inf")
        
        # Perulangan hingga sebanyak jumlah fitur
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # Perulangan setiap fitur yang ada pada dataset
            for threshold in possible_thresholds:
                # Memecah dataset untuk dijadikan sebagai anak kiri dan kanan
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # Melakukan pengecekan apakah anaknya tidak kosong
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # mencari IG pada dataset
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # Mengupdate data ketika iG terkini lebih tinggi dari max IG sebelumnya
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
        
    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def printTree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f"X_{tree.feature_index}<={tree.threshold} ? {tree.info_gain}")
            print("%sleft:" % (indent), end="")
            self.printTree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.printTree(tree.right, indent + indent)


def getHighestKValueAccuracy(K_range, X_train, X_test, y_train, y_test):
  accuracy_vals = []
  maxScore = 0
  maxK = 0
  for i in range(1,K_range):
    model = KNN(i)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    curScore = score(y_pred, X_test,y_test)
    accuracy_vals.append(curScore)
    if curScore > maxScore:
      maxScore = curScore
      maxK = i

  plt.plot(range(1,K_range), accuracy_vals, color='blue', linestyle= 'dashed', marker='x')
  plt.xlabel("Nilai Tetangga")
  plt.ylabel("Akurasi")
  
  return maxK

def printAccuracy(score):
    print("Accuracy: %.2f" % (score*100), "%")
    
def score(y_pred,X_test,y_test):
    return np.sum(y_pred==y_test)/len(X_test)
