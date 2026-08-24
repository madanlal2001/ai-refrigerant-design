"""D-MPNN with PRETRAINING under FROZEN PROTOCOL v2.
Merges: friend's pretrain->finetune idea + our frozen cluster split + our D-MPNN.

Stage 1 (pretrain): estimated critprop data (6168) + Towhee GEMC Tc labels (error<=25 K),
                    EXCLUDING any molecule whose canonical SMILES is in val/test or the 142 set.
Stage 2 (finetune): experimental 1825 train partition; early stop on frozen val.
Eval: frozen test, once. Then deploy on 142.

Run: python dmpnn_pretrain_frozen.py   (GPU strongly recommended)
"""
import json, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# ================= CONFIG =================
TRAIN_FILE   = "all_unique_molecules_Tc_Pc_omega_1825.csv"
TARGET_FILE  = "molecules_144_refrigerant.xlsx"
EST_FILE     = "critprop_data_est_updated_existing_only.csv"        # estimated pretrain data
TOWHEE_FILE  = "towhee_tc_results_combined.csv"                    # GEMC Tc weak labels
TOWHEE_TC_ERROR_CUTOFF = 25.0
MAX_PRETRAIN_ATOMS = 30
OUT = Path("dmpnn_pretrain_results"); OUT.mkdir(exist_ok=True)
TARGETS = ["Tc_K", "Pc_bar", "omega"]; OMEGA_DIAG_RANGE = (-0.3, 1.0)
SEED=42; BATCH=64; HIDDEN=300; DEPTH=3; DROPOUT=.1
PRETRAIN_EPOCHS=60; EPOCHS=300; PATIENCE=30; LR=1e-3; FINETUNE_LR=5e-4
CLUSTER_CUTOFF=0.60; SPLIT_FP_BITS=2048
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def oh(v,c): return [float(v==x) for x in c]+[float(v not in c)]
def af(a):
    f=[]; f+=oh(a.GetAtomicNum(),[1,5,6,7,8,9,14,15,16,17,35,53]); f+=oh(a.GetTotalDegree(),range(6))
    f+=oh(a.GetFormalCharge(),[-2,-1,0,1,2]); f+=oh(int(a.GetChiralTag()),range(4))
    f+=oh(int(a.GetTotalNumHs()),range(5)); f+=oh(int(a.GetHybridization()),[2,3,4,5,6])
    f+=[float(a.GetIsAromatic()),float(a.IsInRing()),a.GetMass()/100]; return np.asarray(f,np.float32)
def bf(b):
    t=[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
    return np.asarray([float(b.GetBondType()==x) for x in t]+[float(b.GetIsConjugated()),float(b.IsInRing())]+oh(int(b.GetStereo()),[0,1,2]),np.float32)
def canon(s):
    m=Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    return Chem.MolToSmiles(m,canonical=True) if m else None

_DIM_MOL=Chem.MolFromSmiles("CC")
ATOM_FDIM=len(af(_DIM_MOL.GetAtomWithIdx(0)))
BOND_FDIM=len(bf(_DIM_MOL.GetBondWithIdx(0)))

class G:
    def __init__(self,s):
        m=Chem.MolFromSmiles(s); self.a=np.stack([af(x) for x in m.GetAtoms()]); src=[];dst=[];b=[];emap={}
        if self.a.shape[1]!=ATOM_FDIM: raise ValueError(f"Atom feature dimension mismatch for {s}: {self.a.shape[1]} != {ATOM_FDIM}")
        for x in m.GetBonds():
            i,j=x.GetBeginAtomIdx(),x.GetEndAtomIdx(); fb=bf(x)
            if len(fb)!=BOND_FDIM: raise ValueError(f"Bond feature dimension mismatch for {s}: {len(fb)} != {BOND_FDIM}")
            for u,v in [(i,j),(j,i)]: emap[u,v]=len(src);src.append(u);dst.append(v);b.append(fb)
        self.src=np.asarray(src,np.int64);self.dst=np.asarray(dst,np.int64)
        self.b=np.stack(b) if b else np.zeros((0,BOND_FDIM),np.float32)
        if self.b.shape[1]!=BOND_FDIM: raise ValueError(f"Bond feature dimension mismatch for {s}: {self.b.shape[1]} != {BOND_FDIM}")
        self.rev=np.asarray([emap[v,u] for u,v in zip(src,dst)],np.int64)

class DS(Dataset):
    def __init__(self,df,scaler=None):
        self.df=df.reset_index(drop=True); self.gs=[G(s) for s in self.df.canonical_smiles]
        y=self.df[TARGETS].to_numpy(np.float32); self.mask=~np.isnan(y)
        if scaler is None:
            mean=np.nanmean(y,0);std=np.nanstd(y,0);std[std==0]=1;scaler=(mean,std)
        self.scaler=scaler; self.y=np.where(self.mask,(y-scaler[0])/scaler[1],0).astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self,i): return self.gs[i],self.y[i],self.mask[i],i

def collate(batch):
    A=[];B=[];S=[];D=[];R=[];M=[];Y=[];K=[];I=[];ao=eo=0
    for mi,(g,y,k,idx) in enumerate(batch):
        if g.a.shape[1]!=ATOM_FDIM: raise ValueError(f"Atom feature dimension mismatch in batch: {g.a.shape[1]} != {ATOM_FDIM}")
        if g.b.shape[1]!=BOND_FDIM: raise ValueError(f"Bond feature dimension mismatch in batch: {g.b.shape[1]} != {BOND_FDIM}")
        A.append(g.a);B.append(g.b);S.extend((g.src+ao).tolist());D.extend((g.dst+ao).tolist())
        R.extend((g.rev+eo).tolist());M.extend([mi]*len(g.a));Y.append(y);K.append(k);I.append(idx)
        ao+=len(g.a);eo+=len(g.src)
    return tuple(torch.tensor(x,dtype=t) for x,t in [(np.concatenate(A),torch.float32),(np.concatenate(B),torch.float32),(S,torch.long),(D,torch.long),(R,torch.long),(M,torch.long),(np.asarray(Y),torch.float32),(np.asarray(K),torch.bool),(I,torch.long)])

class DMPNN(nn.Module):
    def __init__(self,ad,bd):
        super().__init__(); self.ein=nn.Linear(ad+bd,HIDDEN); self.em=nn.Linear(HIDDEN,HIDDEN,bias=False)
        self.aout=nn.Linear(ad+HIDDEN,HIDDEN)
        self.ff=nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(HIDDEN,3))
        self.act=nn.ReLU(); self.drop=nn.Dropout(DROPOUT)
    def forward(self,a,b,s,d,r,m):
        if len(s):
            h0=self.act(self.ein(torch.cat([a[s],b],1))); h=h0
            for _ in range(DEPTH-1):
                inc=torch.zeros((len(a),HIDDEN),device=a.device); inc.index_add_(0,d,h)
                h=self.act(h0+self.em(inc[s]-h[r])); h=self.drop(h)
            agg=torch.zeros((len(a),HIDDEN),device=a.device); agg.index_add_(0,d,h)
        else: agg=torch.zeros((len(a),HIDDEN),device=a.device)
        ha=self.act(self.aout(torch.cat([a,agg],1))); n=int(m.max())+1
        hm=torch.zeros((n,HIDDEN),device=a.device); hm.index_add_(0,m,ha)
        hm/=torch.bincount(m,minlength=n).float().unsqueeze(1).clamp_min(1)
        return self.ff(hm)

def mv(z): return tuple(x.to(DEVICE) for x in z)
def clone_state_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def scaler_tensors(mean,std):
    return torch.as_tensor(mean,dtype=torch.float32).clone(),torch.as_tensor(std,dtype=torch.float32).clone()

# ===== FROZEN SPLIT PROTOCOL (identical across all model scripts) =====
def frozen_cluster_split(canonical_smiles,cutoff=CLUSTER_CUTOFF,fp_bits=SPLIT_FP_BITS,fractions=(0.8,0.1,0.1)):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=fp_bits)
    fps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in canonical_smiles]
    n=len(fps); dists=[]
    for i in range(1,n): dists.extend(1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    clusters=Butina.ClusterData(dists,n,cutoff,isDistData=True)
    names=("train","val","test"); target={k:f*n for k,f in zip(names,fractions)}; cnt={k:0 for k in names}
    split=np.empty(n,object); cid=np.zeros(n,int)
    for c,members in sorted(enumerate(clusters),key=lambda z:(-len(z[1]),z[0])):
        deficit={k:(target[k]-cnt[k])/target[k] for k in names}
        q=max(names,key=lambda k:deficit[k])
        for i in members: split[i]=q; cid[i]=c
        cnt[q]+=len(members)
    return split,cid,fps
# ===== END FROZEN SPLIT PROTOCOL =====

def lossfn(p,y,k):
    return torch.stack([((p[k[:,j],j]-y[k[:,j],j])**2).mean() for j in range(3) if k[:,j].any()]).mean()

@torch.no_grad()
def pred(model,loader):
    model.eval();P=[];Y=[];K=[];I=[]
    for z in loader:
        a,b,s,d,r,m,y,k,i=mv(z)
        P.append(model(a,b,s,d,r,m).cpu().numpy());Y.append(y.cpu().numpy());K.append(k.cpu().numpy());I.append(i.cpu().numpy())
    return np.vstack(P),np.vstack(Y),np.vstack(K),np.concatenate(I)

def metric_dict(y,p):
    e=np.abs(y-p)
    return {"n":int(len(y)),"R2":float(r2_score(y,p)),"RMSE":float(mean_squared_error(y,p)**.5),
            "MAE":float(mean_absolute_error(y,p)),"MedAE":float(np.median(e)),"MaxAE":float(e.max())}

def build_pretrain_df(banned_canonical):
    est=pd.read_csv(EST_FILE).rename(columns={"Tc (K)":"Tc_K","Pc (bar)":"Pc_bar","omega (-)":"omega"})
    est["canonical_smiles"]=est.smiles.map(canon)
    est=est.dropna(subset=["canonical_smiles"])
    est=est[~est.canonical_smiles.isin(banned_canonical)]
    def natoms(s):
        m=Chem.MolFromSmiles(s); return Chem.AddHs(m).GetNumAtoms() if m else 999
    est=est[est.canonical_smiles.map(natoms)<=MAX_PRETRAIN_ATOMS]
    if "omega" not in est.columns: est["omega"]=np.nan
    est=est[["canonical_smiles","Tc_K","Pc_bar","omega"]]
    tw=pd.read_csv(TOWHEE_FILE)
    tw=tw[tw.towhee_tc_error<=TOWHEE_TC_ERROR_CUTOFF]
    tw["canonical_smiles"]=tw.smiles.map(canon)
    tw=tw.dropna(subset=["canonical_smiles"])
    tw=tw[~tw.canonical_smiles.isin(banned_canonical)]
    tw=pd.DataFrame({"canonical_smiles":tw.canonical_smiles,"Tc_K":tw.towhee_tc_mean,"Pc_bar":np.nan,"omega":np.nan})
    pre=pd.concat([est,tw],ignore_index=True)
    pre=pre.drop_duplicates(subset="canonical_smiles",keep="first")
    pre=pre.dropna(subset=TARGETS,how="all").reset_index(drop=True)
    return pre

def train_loop(model,loader,opt,epochs,val_loader=None,patience=None):
    best=1e99;bad=0;state=None;be=0
    for ep in range(1,epochs+1):
        model.train();ls=[]
        for z in loader:
            a,b,s,d,r,m,y,k,_=mv(z);opt.zero_grad()
            l=lossfn(model(a,b,s,d,r,m),y,k);l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(l.item())
        if val_loader is not None:
            p,y,k,_=pred(model,val_loader)
            vl=np.mean([np.mean((p[k[:,j],j]-y[k[:,j],j])**2) for j in range(3) if k[:,j].any()])
            if vl<best-1e-6: best=vl;bad=0;be=ep;state={kk:v.detach().cpu().clone() for kk,v in model.state_dict().items()}
            else: bad+=1
            if ep==1 or ep%10==0: print(f"  epoch {ep:3d} train={np.mean(ls):.5f} val={vl:.5f}")
            if patience and bad>=patience: break
        elif ep==1 or ep%10==0: print(f"  epoch {ep:3d} train={np.mean(ls):.5f}")
    if state is not None: model.load_state_dict(state)
    return model,be

def main():
    df=pd.read_csv(TRAIN_FILE); tg=pd.read_excel(TARGET_FILE)
    df["canonical_smiles"]=df.smiles.map(canon); tg["canonical_smiles"]=tg.SMILES.map(canon)
    if df.canonical_smiles.isna().any() or tg.canonical_smiles.isna().any(): raise ValueError("Invalid SMILES")
    if df.canonical_smiles.duplicated().any() or tg.canonical_smiles.duplicated().any(): raise ValueError("Canonical duplicates detected")

    sp,cid,split_fps=frozen_cluster_split(df.canonical_smiles.tolist())
    df["split"]=sp; df["cluster_id"]=cid
    df.to_csv(OUT/"frozen_split_assignments.csv",index=False)
    tr=df[df.split=="train"];va=df[df.split=="val"];te=df[df.split=="test"]
    print(f"Device={DEVICE} split train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # scaler from experimental TRAIN only (frozen)
    mean=np.nanmean(tr[TARGETS].to_numpy(float),0).astype(np.float32)
    std=np.nanstd(tr[TARGETS].to_numpy(float),0).astype(np.float32);std[std==0]=1;scaler=(mean,std)

    # ---- Stage 1: pretraining (leakage-safe) ----
    banned=set(va.canonical_smiles)|set(te.canonical_smiles)|set(tg.canonical_smiles)
    pre=build_pretrain_df(banned)
    print(f"Pretraining rows after exclusions: {len(pre)} (labels: {pre[TARGETS].notna().sum().to_dict()})")
    pre_ds=DS(pre,scaler); pre_loader=DataLoader(pre_ds,BATCH,True,collate_fn=collate)
    model=DMPNN(ATOM_FDIM,BOND_FDIM).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-6)
    print("Stage 1: pretraining")
    model,_=train_loop(model,pre_loader,opt,PRETRAIN_EPOCHS)
    pretrained_state=clone_state_dict(model)
    mean_t,std_t=scaler_tensors(mean,std)
    torch.save({"state_dict":pretrained_state,"mean":mean_t,"std":std_t},OUT/"dmpnn_pretrained.pt")

    # ---- Stage 2: fine-tune on experimental train, early stop on frozen val ----
    model=DMPNN(ATOM_FDIM,BOND_FDIM).to(DEVICE); model.load_state_dict(pretrained_state)
    td,vd=DS(tr,scaler),DS(va,scaler)
    tl=DataLoader(td,BATCH,True,collate_fn=collate); vl=DataLoader(vd,BATCH,False,collate_fn=collate)
    opt=torch.optim.Adam(model.parameters(),lr=FINETUNE_LR,weight_decay=1e-6)
    print("Stage 2: fine-tuning")
    model,be=train_loop(model,tl,opt,EPOCHS,vl,PATIENCE)

    # ---- one-time frozen test evaluation ----
    ds=DS(te,scaler);loader=DataLoader(ds,BATCH,False,collate_fn=collate)
    p,y,k,ids=pred(model,loader); pu=p*std+mean; yu=y*std+mean
    metrics={}
    for j,t in enumerate(TARGETS):
        q=k[:,j]; metrics[t]=metric_dict(yu[q,j],pu[q,j])
        if t=="omega":
            lo,hi=OMEGA_DIAG_RANGE; dm=q&(yu[:,j]>=lo)&(yu[:,j]<=hi)
            metrics[f"omega_diag[{lo},{hi}]"]=metric_dict(yu[dm,j],pu[dm,j])
    print(json.dumps(metrics,indent=2))
    json.dump({"best_epoch":be,"metrics":metrics},open(OUT/"test_metrics.json","w"),indent=2)

    tr_pos=np.where(df.split.values=="train")[0]; fps_train=[split_fps[i] for i in tr_pos]
    to=te.iloc[ids][["name","smiles","canonical_smiles","source","split","cluster_id"]].copy()
    te_pos=np.where(df.split.values=="test")[0][ids]
    to["max_tanimoto_to_train"]=[max(DataStructs.BulkTanimotoSimilarity(split_fps[i],fps_train)) for i in te_pos]
    for j,t in enumerate(TARGETS):
        act=np.where(k[:,j],yu[:,j],np.nan);pr=pu[:,j]
        to[f"{t}_actual"]=act;to[f"{t}_pred"]=pr;to[f"{t}_residual"]=act-pr;to[f"{t}_abs_error"]=np.abs(act-pr)
    to.to_csv(OUT/"test_predictions.csv",index=False)
    torch.save({"state_dict":clone_state_dict(model),"mean":mean_t,"std":std_t,"best_epoch":int(be)},OUT/"dmpnn_pretrain_eval.pt")

    # ---- deployment: pretrain-init + fine-tune on ALL experimental labels for be epochs ----
    final=DMPNN(ATOM_FDIM,BOND_FDIM).to(DEVICE); final.load_state_dict(pretrained_state)
    all_ds=DS(df,scaler);al=DataLoader(all_ds,BATCH,True,collate_fn=collate)
    opt=torch.optim.Adam(final.parameters(),lr=FINETUNE_LR,weight_decay=1e-6)
    final,_=train_loop(final,al,opt,max(be,1))
    fake=pd.DataFrame({"canonical_smiles":tg.canonical_smiles,"Tc_K":np.nan,"Pc_bar":np.nan,"omega":np.nan})
    tl2=DataLoader(DS(fake,scaler),BATCH,False,collate_fn=collate)
    tp,_,_,_=pred(final,tl2); tp=tp*std+mean
    for j,t in enumerate(TARGETS): tg[f"{t}_pred"]=tp[:,j]
    tg["exact_training_overlap"]=tg.canonical_smiles.isin(set(df.canonical_smiles))
    tg.to_excel(OUT/"dmpnn_pretrain_142_predictions.xlsx",index=False)
    torch.save({"state_dict":clone_state_dict(final),"mean":mean_t,"std":std_t},OUT/"dmpnn_pretrain_final.pt")
    print(f"Done. best_epoch={be} outputs -> {OUT.resolve()}")

if __name__=="__main__":
    main()
