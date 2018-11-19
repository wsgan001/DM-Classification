def DataGen():
    start_work_age = random.randint(22, 40)
    exp            = random.randint( 0, 25)
    age = start_work_age + exp
    edu = random.randint(0, 2)
    performance = random.randint(0, 2)
    reward = random.randint( 0, 10)
    relationship = random.randint(0, 2)
    label = 1
    if relationship >= 1:
        if performance >= 1:
            if edu >= 2:
                label = 1;
            else:
                if reward >= 4:
                    label = 1;
                else:
                    label = -1;
        else:
            if exp >= 15:
                label = 1;
            else:
                label = -1;
    else:
        label = -1;

    return [age, exp, edu, performance, reward, relationship], label
        

def GenBatch(n=10000):
    x_batch = []
    t_batch = []
    for _ in range(n):
        feature, promotion = DataGen()
        x_batch.append(feature)
        t_batch.append(promotion)
    return np.array(x_batch), np.array(t_batch)

def List2VecInt(_list):
    vec_i = CWrapper.newVectorInt()
    for item in _list:
        CWrapper.VectorIntPush(vec_i, c_int(item))
    return vec_i

def VecInt2NP(_vec):
    n = CWrapper.VectorIntLen(_vec)
    res = []
    for i in range(n):
        res.append(CWrapper.VectorIntRead(_vec, i))
    return np.array(res)

def TwoDList2VecFea(_list):
    vec_f = CWrapper.newVectorFeature()
    for item in _list:
        CWrapper.VectorFeaturePush(vec_f, List2Feature(item))
    return vec_f

def VecFea2TwoDNP(_vec):
    n = CWrapper.VectorFeatureLen(_vec)
    res = []
    for i in range(n):
        _f = CWrapper.VectorFeatureRead(_vec, i)
        res.append(Feature2NP(_f))
    return np.array(res)

def List2Feature(_list):
    feature = CWrapper.newFeature(len(_list))
    for i, item in enumerate(_list):
        CWrapper.FeatureSet(feature, i, c_double(item))
    return feature
    
def Feature2NP(_f):
    res = []
    n = CWrapper.FeatureLen(_f)
    for i in range(n):
        res.append((CWrapper.FeatureRead(_f, i)))
    return np.array(res)

def calAccuracy(target, pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for t, p in zip(target, pred):
        if t == 1:
            if p == t:
                TP += 1
            else:
                FN += 1
        else:
            if p == t:
                TN += 1
            else:
                FP += 1

    acc       = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    return acc, precision, recall

if __name__ == '__main__':
    import random
    import numpy as np
    from ctypes import *
    from sklearn.svm import SVC
    CWrapper = CDLL('./lib/libCWrapper.so')
    CWrapper.FeatureRead.restype = c_double

    print('========== Decision Tree ==========')
    # Build training data
    train_x, train_t = GenBatch(10000)
    c_train_x = TwoDList2VecFea(train_x)
    c_train_t = List2VecInt(train_t)

    # Build testing data
    test_x, test_t = GenBatch(10000)
    c_test_x = TwoDList2VecFea(test_x)
    c_test_t = List2VecInt(test_t)

    # Build decision tree model
    DT = CWrapper.newDecisionTree(c_double(.90), 4)

    # Training
    CWrapper.fit(DT, c_train_x, c_train_t)

    # Print Decision Tree
    CWrapper.printTree(DT)

    # Calculate accuracy
    train_pred = VecInt2NP(CWrapper.predict(DT, c_train_x))
    acc, pre, rec = calAccuracy(train_t, train_pred)
    print('----- Training -----')
    print('Accuracy : {:0.5f}'.format(acc))
    print('Precision: {:0.5f}'.format(pre))
    print('Recall   : {:0.5f}'.format(rec))
    print()
    test_pred  = VecInt2NP(CWrapper.predict(DT, c_test_x))
    acc, pre, rec = calAccuracy(test_t, test_pred)
    print('----- Testing ------')
    print('Accuracy : {:0.5f}'.format(acc))
    print('Precision: {:0.5f}'.format(pre))
    print('Recall   : {:0.5f}'.format(rec))

    print('\n========== Support Vector Machine ==========')
    # Build SVC model
    svc = SVC(gamma='scale')

    # Training
    svc.fit(train_x, train_t)

    # Calculate accuracy
    train_pred = svc.predict(train_x)
    acc, pre, rec = calAccuracy(train_t, train_pred)
    print('----- Training -----')
    print('Accuracy : {:0.5f}'.format(acc))
    print('Precision: {:0.5f}'.format(pre))
    print('Recall   : {:0.5f}'.format(rec))
    print()
    test_pred  = svc.predict(test_x)
    acc, pre, rec = calAccuracy(test_t, test_pred)
    print('----- Testing ------')
    print('Accuracy : {:0.5f}'.format(acc))
    print('Precision: {:0.5f}'.format(pre))
    print('Recall   : {:0.5f}'.format(rec))
