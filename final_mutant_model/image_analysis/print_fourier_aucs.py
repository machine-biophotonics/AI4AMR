import pandas as pd, numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("output_all_plates/all_plates_features_fourier.csv")
feats = ["mean","std","snr","entropy","p1","p99","median"]
plates = ["P1","P2","P3","P4","P5","P6"]
prefix = "fourier_"

for region in ["full", "center1128", "center224"]:
    cols = [prefix + region + "_mp_" + f for f in feats]
    X = df[cols].values
    y = (df["type"] == "drug").astype(int).values
    
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        clf = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
        clf.fit(scaler.fit_transform(X[tr]), y[tr])
        score = clf.predict_proba(scaler.transform(X[te]))[:,1]
        fpr, tpr, _ = roc_curve(y[te], score)
        aucs.append(auc(fpr, tpr))
    print("within {}: {:.3f} +/- {:.3f}".format(region, np.mean(aucs), np.std(aucs)))
    
    xaucs = []
    for fi, test in enumerate(plates):
        val = plates[(fi+1)%6]
        train = [p for p in plates if p not in (test, val)]
        X_tr = df.loc[df["plate"].isin(train), cols].values
        y_tr = (df.loc[df["plate"].isin(train), "type"] == "drug").astype(int).values
        X_te = df.loc[df["plate"] == test, cols].values
        y_te = (df.loc[df["plate"] == test, "type"] == "drug").astype(int).values
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
        clf.fit(scaler.transform(X_tr), y_tr)
        score = clf.predict_proba(scaler.transform(X_te))[:,1]
        fpr, tpr, _ = roc_curve(y_te, score)
        xaucs.append(auc(fpr, tpr))
    print("cross {}: {:.3f} +/- {:.3f}".format(region, np.mean(xaucs), np.std(xaucs)))
    print()
