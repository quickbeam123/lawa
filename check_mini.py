#!/usr/bin/env python3

import sys

from dataclasses import dataclass
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text

@dataclass
class Datum:
    sizeKb: int
    loss: float
    took: float
    memMb: int
    sort: int
    symbol: int
    clause: int
    term: int
    var: int
    gage: int
    gweight: int
    box: int

def load_data(data,filename):
  with open(filename) as f:
    # None-ing is just for inputs which don't contain these values
    sort = None
    symbol = None
    term = None
    clause = None
    var = None
    gage = None
    gweight = None
    box = None

    for line in f:
      spl = line.split()
      if spl[0] == "Input:":
        inp = spl[1]
      elif spl[0] == "Ofsize:":
        sizeKb = int(spl[1])
      elif spl[0] == "sort":
        sort = int(spl[-1])
      elif spl[0] == "symbol":
        symbol = int(spl[-1])
      elif spl[0] == "clause":
        clause = int(spl[-1])
      elif spl[0] == "term":
        term = int(spl[-1])
      elif spl[0] == "var":
        var = int(spl[-1])
      elif spl[0] == "Gage-height:":
        gage = int(spl[-1])
      elif spl[0] == "Gweight-height:":
        gweight = int(spl[-1])
      elif spl[0] == "Box:":
        box = int(spl[-1])
      elif spl[0] == "Loss:":
        loss = float(spl[1])
      elif spl[0] == "Took:":
        took = float(spl[1])
      elif spl[0] == "Peak":
        memMb = int(spl[-2])
        data[inp] = Datum(sizeKb,loss,took,memMb,sort,symbol,clause,term,var,gage,gweight,box)
  return data

def show_maxes(data):
  max_took = 0
  max_took_prob = None
  max_memMb = 0
  max_memMb_prob = None
  for prob,d in data.items():
    if d.took > max_took:
      max_took = d.took
      max_took_prob = prob
    if d.memMb > max_memMb:
      max_memMb = d.memMb
      max_memMb_prob = prob

  print("max_took",max_took,max_took_prob)
  print("  ",data[max_took_prob])
  print("max_memMb",max_memMb,max_memMb_prob)
  print("  ",data[max_memMb_prob])

if __name__ == "__main__":
  # see mini.py for info

  data_train = {}
  for train_data in sys.argv[1:]:
    load_data(data_train,train_data)

  print("So far",len(data_train),"training data")

  import matplotlib.pyplot as plt
  fig, ax1 = plt.subplots(figsize=(6,6))

  # cond = lambda d: d.memMb > 15000 or d.took > 800
  cond = lambda d: d.box > 175000 or d.sizeKb > 150000

  # scatter "took" against "memMb"
  if True:
    X1s = []
    Y1s = []
    X2s = []
    Y2s = []
    for prob,d in data_train.items():
      if cond(d):
        X2s.append(d.took)
        Y2s.append(d.memMb)
      else:
        X1s.append(d.took)
        Y1s.append(d.memMb)

    ax1.scatter(X1s,Y1s,s=1)
    ax1.scatter(X2s,Y2s,s=1)

    plt.xlabel("took (s)")
    plt.ylabel("mem (MB)")

  if False:
    # plot memMb vs sizeKb
    Xs = []
    Ys = []
    for prob,d in data_train.items():
      Xs.append(d.memMb)
      Ys.append(d.symbol)
      if d.symbol > 25000:
        print(prob,d)

    ax1.scatter(Xs,Ys,s=1)

    plt.xlabel("train - memMb")
    plt.ylabel("train - sizeKb")

    # ax1.set_xlim([1, 510])
    # ax1.set_ylim([1, 510])

  # use a simple predictor to predict memMb > 10000 while using any feature in the data except "took" and "mdemMb"
  if False:
    # Prepare the data for classification
    features = []
    target = []
    for prob, d in data_train.items():
      if not cond(d):
        features.append([d.sizeKb, d.sort, d.symbol, d.clause, d.term, d.var, d.gage, d.gweight, d.box])
        target.append(1 if d.took > 600 else 0)

    features = np.array(features)
    target = np.array(target)

    # Create and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=1)
    clf.fit(features, target)

    # Print the feature importances
    print("Feature importances:", clf.feature_importances_)

    # show the random forest classifier
    tree_rules = export_text(clf.estimators_[0])
    print(tree_rules)

    # Predict the probabilities for the training data
    predictions = clf.predict_proba(features)[:, 1]


  if False:
    max_took = 0
    # plot took vs box
    Xs = []
    Ys = []
    for prob,d in data_train.items():
      Xs.append(d.took)
      Ys.append(d.box)

      if d.box > 175000:
        print(prob,d)
      else:
        max_took = max(max_took,d.took)

    ax1.scatter(Xs,Ys,s=1)

    plt.xlabel("train - took")
    plt.ylabel("train - box")

    # ax1.set_xlim([1, 510])
    # ax1.set_ylim([1, 510])

    print("Clipped max_took",max_took)

  if False:
    # Prepare the data for linear regression
    features = []
    target = []
    for prob, d in data_train.items():
      features.append([d.sizeKb, d.sort, d.symbol, d.clause, d.term, d.var, d.gage, d.gweight, d.box])
      target.append(d.took)

    # coef's for predicting memMb
    # sizeKb  *  0.15392619
    # sort    *  1.76768645
    # symbol  * -0.01807480
    # clause  *  0.04978758
    # term    * -0.00628216
    # var     * -0.00022339
    # gage    *  1.79248067
    # gweight * -1.15508931
    # box     * -0.04973371

    features = np.array(features)
    target = np.array(target)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(features, target)

    # Print the coefficients
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Predict memMb for the training data
    predictions = model.predict(features)

    # Plot the predicted vs actual memMb
    plt.scatter(target, predictions, s=1)
    # plt.xlim([0, 14000])
    # plt.ylim([0, 14000])
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.xlabel("Actual took")
    plt.ylabel("Predicted took")
    # plt.title("Linear Regression: Actual vs Predicted took")
    plt.savefig("linear_regression_took.pdf", format="pdf", bbox_inches="tight")
    plt.close()

  # scatter took against took
  if False:
    Xs = []
    Ys = []
    for prob,d in data_train.items():
      Xs.append(data_eval[prob].took)
      Ys.append(d.took)

      if d.took > 150:
        print(prob,d)

    ax1.scatter(Xs,Ys,s=1)

    plt.xlabel("took - eval")
    plt.ylabel("took - train")

    ax1.set_xlim([1, 510])
    ax1.set_ylim([1, 510])

  # scatter mem against mem
  if False:
    Xs = []
    Ys = []
    for prob,d in data_train.items():
      Xs.append(data_eval[prob].memMb)
      Ys.append(d.memMb)

    ax1.scatter(Xs,Ys,s=1)

    plt.xlabel("mem - eval")
    plt.ylabel("mem - train")

    # ax1.set_xlim([1, 510])
    # ax1.set_ylim([1, 510])

  # scatter mem against mem
  if False:
    Xs = []
    Ys = []
    for prob,d in data_train.items():
      Xs.append(data_eval[prob].memMb)
      Ys.append(d.memMb)

    ax1.scatter(Xs,Ys,s=1)

    plt.xlabel("mem - eval")
    plt.ylabel("mem - train")

    # ax1.set_xlim([1, 450])
    # ax1.set_ylim([1, 450])

  plt.savefig("mini_scatter.pdf",format="pdf", bbox_inches="tight")
  plt.close(fig)


