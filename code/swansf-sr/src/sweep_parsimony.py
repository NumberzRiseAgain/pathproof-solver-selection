"""Calibrate parsimony_coefficient against the |Gini| fitness scale.
Run on the SYNTHETIC surrogate only, before any real partition is loaded.
Selection criterion is fixed in advance: highest validation TSS, and among
configurations within 0.02 of the best, the smallest expression."""
import json, sys, numpy as np
sys.path.insert(0,'.')
import swan_io
from features import extract, TrainOnlyScaler
from gp_search import multi_start
from sklearn.model_selection import train_test_split

tr = swan_io.make_synthetic_partition(n=1400, seed=11)
F,names = extract(tr.X,'mvts',tr.params)
itr,iva = train_test_split(np.arange(len(tr.y)),test_size=.3,stratify=tr.y,random_state=7)
sc = TrainOnlyScaler().fit(F[itr]); A,B = sc.transform(F[itr]), sc.transform(F[iva])
ytr,yva = tr.y[itr], tr.y[iva]

rows=[]
for pc in (0.0, 0.00005, 0.0002, 0.0008, 0.002):
    runs = multi_start(A,ytr,B,yva,names,seeds=(0,1,2),
                       generations=20, population_size=1500, parsimony=pc)
    best=runs[0]
    rows.append({"parsimony":pc,"val_tss":round(best["val_tss"],4),
                 "nodes":int(best["length"]),
                 "all_val_tss":[round(r["val_tss"],4) for r in runs],
                 "all_nodes":[int(r["length"]) for r in runs],
                 "expr":best["expr"][:110]})
    print(f"pc={pc:<8} valTSS={rows[-1]['val_tss']:.4f} nodes={rows[-1]['nodes']:<3} "
          f"seeds={rows[-1]['all_val_tss']} {rows[-1]['expr']}", flush=True)

best_tss=max(r["val_tss"] for r in rows)
elig=[r for r in rows if r["val_tss"]>=best_tss-0.02]
pick=min(elig,key=lambda r:r["nodes"])
print("\nSELECTED parsimony_coefficient =",pick["parsimony"],
      "  val_tss",pick["val_tss"],"  nodes",pick["nodes"])
json.dump({"synthetic":True,"sweep":rows,"selected":pick},
          open("../results/parsimony_sweep.json","w"),indent=2)
