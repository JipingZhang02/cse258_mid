# %%
import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model
import random
import statistics

z = gzip.open("train.json.gz")

dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

z.close()

# %%
answers = {}

# %%
dataset[1]

# %%
# def MSE(y, ypred):
#     if isinstance(y,numpy.ndarray):
#         y = y.reshape((-1,))
#     if isinstance(ypred,numpy.ndarray):
#         ypred = ypred.reshape((-1,))
#     if len(y)!=len(ypred):
#         raise ValueError("len(y) don't equal len(ypred)")
#     sq_err_sum = 0.0
#     for yl,yp in zip(y,ypred):
#         sq_err_sum+=(yl-yp)**2
#     return sq_err_sum/len(y)

# def MAE(y, ypred):
#     if isinstance(y,numpy.ndarray):
#         y = y.reshape((-1,))
#     if isinstance(ypred,numpy.ndarray):
#         ypred = ypred.reshape((-1,))
#     if len(y)!=len(ypred):
#         raise ValueError("len(y) don't equal len(ypred)")
#     abs_err_sum = 0.0
#     for yl,yp in zip(y,ypred):
#         abs_err_sum+=abs(yl-yp)
#     return abs_err_sum/len(y)

def MSE(y, ypred):
    assert isinstance(y,numpy.ndarray)
    assert isinstance(ypred,numpy.ndarray)
    assert y.shape==ypred.shape
    sq_err = (y-ypred)**2
    return numpy.mean(sq_err)

def MAE(y, ypred):
    assert isinstance(y,numpy.ndarray)
    assert isinstance(ypred,numpy.ndarray)
    assert y.shape==ypred.shape
    abs_err = numpy.abs(y-ypred)
    return numpy.mean(abs_err)

# %%
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])

# %%
def feat1(d):
    return [1.0,float(d['hours'])]

X = list(feat1(d) for d in dataset)
X = numpy.array(X)
y = list(float(len(d['text'])) for d in dataset)
y = numpy.array(y)
y = y.reshape((-1,1))
mod = linear_model.LinearRegression()
mod.fit(X,y)

# %%
theta_1 = mod.coef_[0][1]
ypred = mod.predict(X)
mse_q1 = MSE(y,ypred)
answers['Q1'] = [theta_1, mse_q1]

# %%
answers

# %%
def lin_reg_pipeline(get_xy_func):
    Xs = list()
    ys = list()
    for d in dataset:
        x1,y1 = get_xy_func(d)
        Xs.append(x1)
        ys.append(y1)
    Xs = numpy.array(Xs,dtype=float)
    if len(Xs.shape)==1:
        Xs = Xs.reshape((-1,1))
    ys = numpy.array(ys,dtype=float)
    print("average of y:"+str(numpy.average(ys)))
    print("variance of y:"+str(numpy.var(ys,ddof=1)))
    if len(ys.shape)==1:
        ys = ys.reshape((-1,1))
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(Xs,ys)
    ypred = model.predict(Xs)
    return model,MSE(y,ypred),MAE(y,ypred)

# %%
def t_transform(t):
    return math.log2(t+1)

# %%
all_hours = list(d['hours'] for d in dataset)

def calculate_median(l):
    sorted_l = sorted(l)
    list_length = len(sorted_l)   
    if list_length == 0:
        return None
    if list_length % 2 == 1:
        median = sorted_l[list_length // 2]
    else:
        m1 = sorted_l[list_length // 2 - 1]
        m2 = sorted_l[list_length // 2]
        median = (m1 + m2) / 2
    return median

median_play_time = calculate_median(all_hours)

def get_xy_2(d):
    l = float(len(d['text']))
    t = float(d['hours'])
    return [1.0,t,t_transform(t),math.sqrt(t),int(t>median_play_time)],l

mod2,mse2,mae2 = lin_reg_pipeline(get_xy_2)
answers['Q2'] = mse2

# %%
def get_xy_3(d):
    l = float(len(d['text']))
    t = float(d['hours'])
    x1 = list()
    x1.append(1.0)
    for t_ref in [1,5,10,100,1000]:
        x1.append(int(t>t_ref))
    return x1,l

mod3,mse3,mae3 = lin_reg_pipeline(get_xy_3)
answers['Q3'] = mse3

# %%
# def get_xy_4(d):
#     l = float(len(d['text']))
#     t = float(d['hours'])
#     return [1.0,l],t

# mod4,mse4,mae4 = lin_reg_pipeline(get_xy_4)
# answers['Q4'] = [mse4, mae4, "mae is better, because review_len and time_played are not so relevant, the mse of the model is extremly big"]

# %%
def feat4(d):
    return [1.0,float(len(d['text']))]

X = [feat4(d) for d in dataset]
X = numpy.array(X)
y = [[float(d['hours'])] for d in dataset]
y = numpy.array(y)

mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

mse = MSE(y,predictions)
mae = MAE(y,predictions)
answers['Q4'] = [mse, mae, "mae is better, because review_len and time_played are not so relevant, the mse of the model is extremly big"]

# %%
y_trans = numpy.vectorize(t_transform)(y)

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
mse_trans = MSE(y_trans,predictions_trans)

# %%
mse_untrans =  MSE(y,2**predictions_trans-1)

# %%
answers['Q5'] = [mse_trans, mse_untrans]

# %%
def get_1hot(l):
    res = list()
    for _ in range(l):
        res.append(0)
    return res
        

def feat6(d):
    h = float(d['hours'])
    int_h = int(h)
    if int_h>=99:
        int_h=99
    res = get_1hot(100)
    res[int_h]=1.0
    return res
    
X = [feat6(d) for d in dataset]
X = numpy.array(X)
y = [len(d['text']) for d in dataset]
y = numpy.array(y)
y=y.reshape((-1,1))
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
models = {}
mses = {}
bestC = None

for c in [1, 10, 100, 1000, 10000]:
    model = linear_model.Ridge(alpha=float(c))
    model.fit(Xtrain,ytrain)
    models[c] = model
    mse_valid = MSE(yvalid,model.predict(Xvalid))
    mses[c] = mse_valid
    if bestC==None:
        bestC = c
    else:
        if mse_valid<mses[bestC]:
            bestC = c

# %%
mse_valid = mses[bestC]
mse_test = MSE(ytest,model.predict(Xtest))
answers['Q6'] = [bestC, mse_valid, mse_test]

# %%
for d in dataset:
    d['hours_transformed'] = t_transform(d['hours'])
times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)

# %%
less_than_1h_cnt = 0
for d in dataset:
    if d['hours']<1.0:
        less_than_1h_cnt+=1

# %%
answers['Q7'] = [median, less_than_1h_cnt]

# %%
def feat8(d):
    return [1.0,float(len(d['text']))]
X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]

# %%
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions

# %%
def get_performance_info(y,predictions):
    y = numpy.array(y,dtype=int)
    y = y.reshape((-1,))
    predictions = numpy.array(predictions,dtype=int)
    predictions = predictions.reshape((-1,))
    # print(y_actual)
    # print(y_predict)
    TP = numpy.sum((y == 1) & (predictions == 1))
    FP = numpy.sum((y == 0) & (predictions == 1))
    TN = numpy.sum((y == 0) & (predictions == 0))
    FN = numpy.sum((y == 1) & (predictions == 0))
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)
    BER = 1 - (0.5 * (TPR + TNR))
    return TP,FP,TN,FN,TPR, FPR, TNR, FNR, BER

# %%
TP,FP,TN,FN,TPR, FPR, TNR, FNR, BER = get_performance_info(y,predictions)
answers['Q8'] = [TP, TN, FP, FN, BER]

# %%
answers

# %%
# TODO : Q9 Q10

# %%
# Q11 code
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
userMedian = defaultdict(list)
itemMedian = defaultdict(list)

dataTrain[0]

for d in dataTrain:
    uid,item_id,h = d["userID"],d["gameID"],d["hours"]
    userMedian[uid].append(h)
    itemMedian[item_id].append(h)



# %%
for u in userMedian:
    userMedian[u] = statistics.median(userMedian[u])

for i in itemMedian:
    itemMedian[i] = statistics.median(itemMedian[i])

# %%
answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]

# %%
all_times_train = list(d['hours'] for d in dataTrain)
global_median = statistics.median(all_times_train)

def f12(u,i):
    if i in itemMedian:
        return int(itemMedian[i]>global_median)
    if u in userMedian:
        return int(userMedian[u]>global_median)
    return 0

preds = [f12(d['userID'], d['gameID']) for d in dataTest]
y = [int(d['hours']>global_median) for d in dataTest]
correct_cnt = 0
for yl,yp in zip(y,preds):
    if yl==yp:
        correct_cnt+=1
accuracy = correct_cnt/len(y)
answers['Q12'] = accuracy

# %%


# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
rating_dict = {}

for d in dataset:
    user,item,tt = d['userID'], d['gameID'],d['hours_transformed']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    rating_dict[(user,item)]=tt

# %%
def mostSimilar(i, func, N):
    item_with_sim = list()
    for j in usersPerItem:
        if j==i:
            continue
        sim_j=func(i,j)
        item_with_sim.append((sim_j,j))
    item_with_sim.sort(key=lambda tup:tup[0],reverse=True)
    return item_with_sim[:N]

# %%
def jaccard(i,j):
    si = usersPerItem[i]
    sj = usersPerItem[j]
    return len(si.intersection(sj))/len(si.union(sj))

def cos_sim_14(i,j):
    si = usersPerItem[i]
    sj = usersPerItem[j]
    i_norm = math.sqrt(len(si))
    j_norm = math.sqrt(len(sj))
    numerator = 0
    for shared_u in si.intersection(sj):
        numerator+=int(rating_dict[(shared_u,i)]>global_median)*int(rating_dict[(shared_u,j)]>global_median)
    return numerator/(i_norm*j_norm)

def cos_sim_15(i,j):
    si = usersPerItem[i]
    sj = usersPerItem[j]
    i_norm = math.sqrt(sum(rating_dict[(u,i)]**2 for u in si))
    j_norm = math.sqrt(sum(rating_dict[(u,j)]**2 for u in sj))
    numerator = 0.0
    for shared_u in si.intersection(sj):
        numerator+=rating_dict[(shared_u,i)]*rating_dict[(shared_u,j)]
    return numerator/(i_norm*j_norm)

ms = mostSimilar(dataset[0]['gameID'], jaccard, 10)
answers['Q13'] = [ms[0][0], ms[-1][0]]
ms = mostSimilar(dataset[0]['gameID'], cos_sim_14, 10)
answers['Q14'] = [ms[0][0], ms[-1][0]]
ms = mostSimilar(dataset[0]['gameID'], cos_sim_15, 10)
answers['Q15'] = [ms[0][0], ms[-1][0]]

# %%
answers

# %%
f = open("answers_midterm.txt", 'w+')
f.write(str(answers) + '\n')
f.close()


