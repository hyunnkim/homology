import numpy as np
import pandas as pd
import yfinance as yf
from ripser import ripser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def log_returns(prices):
    prices = np.asarray(prices).reshape(-1)
    return pd.Series(np.log(prices)).diff()

def takens_embedding(x, m, tau):
    x = np.asarray(x).reshape(-1)
    n = len(x)
    k = n - (m - 1) * tau
    if k <= 1:
        return np.empty((0, m))
    X = np.zeros((k, m))
    for j in range(m):
        s = (m - 1 - j) * tau
        X[:, j] = x[s:s + k]
    return X

def persistence_entropy(d):
    if d.size == 0:
        return 0.0
    d = d[np.isfinite(d[:, 1])]
    if d.size == 0:
        return 0.0
    p = np.maximum(d[:, 1] - d[:, 0], 0.0)
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def tda_features(w, m, tau):
    emb = takens_embedding(w, m, tau)
    if emb.shape[0] < 5:
        return np.zeros(9)
    dgms = ripser(emb, maxdim=1)["dgms"]
    d0 = dgms[0]
    d1 = dgms[1]

    def stats(d):
        if d.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        d = d[np.isfinite(d[:, 1])]
        if d.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        p = np.maximum(d[:, 1] - d[:, 0], 0.0)
        q = np.quantile(p, 0.75)
        return float(p.max()), float(p.mean()), float(p.sum()), float((p > q).sum())

    a0, b0, c0, d0c = stats(d0)
    a1, b1, c1, d1c = stats(d1)

    return np.array([
        a0, b0, c0, d0c,
        a1, b1, c1, d1c,
        persistence_entropy(d0) + persistence_entropy(d1)
    ])

def build_dataset(prices, W=80, m=5, tau=1):
    r = log_returns(prices).dropna()
    rv = r.values

    X = []
    y = []
    idx = []

    for t in range(W, len(rv) - 1):
        X.append(tda_features(rv[t - W:t], m, tau))
        y.append(1 if rv[t + 1] > 0 else 0)
        idx.append(r.index[t])

    X = pd.DataFrame(X, index=idx)
    y = pd.Series(y, index=idx, name="y")
    return X.join(y)

def walk_forward(prices, df, train_len=600):
    prices = np.asarray(prices).reshape(-1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    X = df.drop(columns="y").values
    y = df["y"].values
    dates = df.index

    probs = []
    pos = []

    for i in range(train_len, len(df)):
        model.fit(X[i - train_len:i], y[i - train_len:i])
        p = model.predict_proba(X[i:i+1])[0, 1]
        probs.append(p)
        pos.append(1.0 if p > 0.55 else -1.0 if p < 0.45 else 0.0)

    out_idx = dates[train_len:]

    probs = pd.Series(probs, index=out_idx)
    pos = pd.Series(pos, index=out_idx)

    r = log_returns(prices).reindex(out_idx)
    strat = pos.shift(1).fillna(0) * r
    equity = (1 + strat.fillna(0)).cumprod()

    return pd.DataFrame({
        "p_up": probs,
        "position": pos,
        "return": r,
        "strategy_return": strat,
        "equity": equity
    })

if __name__ == "__main__":
    data = yf.download("SPY", period="10y", auto_adjust=True, progress=False)
    close = data["Close"].values

    df = build_dataset(close)
    bt = walk_forward(close, df)

    print(bt.tail())
    print(bt["equity"].iloc[-1])
