"""Download pinned Cora files, verify bytes, then assemble the fixed split."""
import argparse,hashlib,json,pickle,urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import networkx as nx
ROOT=Path(__file__).resolve().parents[1]
COMMIT='39a4089fe72ad9f055ed6fdb9746abdcfebc4d81'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/cora')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'data/processed/cora')
    args=ap.parse_args();raw=args.raw_dir;out=args.output_dir
    expected=json.loads((ROOT/'configs/cora_raw_sha256.json').read_text())
    raw.mkdir(parents=True,exist_ok=True)
    for name,wanted in expected.items():
        p=raw/name
        if not p.exists():
            url=f'https://raw.githubusercontent.com/tkipf/gcn/{COMMIT}/gcn/data/{name}'
            data=urllib.request.urlopen(url,timeout=90).read()
            # Original Windows checkout used CRLF for this one text file.
            # Normalize only line endings before checking its original SHA-256.
            if name=='ind.cora.test.index':data=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
            if hashlib.sha256(data).hexdigest()!=wanted:raise ValueError('Download checksum mismatch: '+name)
            with p.open('xb') as f:f.write(data)
        if digest(p)!=wanted:raise ValueError('Raw checksum mismatch: '+name)
    # Only deserialize the eight pinned, checksum-verified source files.
    data={}
    for name in ['x','y','tx','ty','allx','ally','graph']:
        with (raw/('ind.cora.'+name)).open('rb') as f:data[name]=pickle.load(f,encoding='latin1')
    index=np.loadtxt(raw/'ind.cora.test.index',dtype=np.int64)
    x=sp.vstack([data['allx'],data['tx']]).tolil()
    y=np.vstack([data['ally'],data['ty']])
    x[index,:]=data['tx'];y[index,:]=data['ty'];x=x.tocsr().astype(np.float32)
    assert x.shape==(2708,1433) and y.shape==(2708,7)
    assert np.array_equal(np.sort(index),np.arange(1708,2708))
    graph=nx.from_dict_of_lists(data['graph'])
    assert set(graph)==set(range(2708)) and graph.number_of_edges()==5278 and nx.number_of_selfloops(graph)==0
    a=sp.csr_matrix(nx.to_scipy_sparse_array(graph,nodelist=range(2708),dtype=np.float32,format='csr'))
    assert a.nnz==10556 and (a!=a.T).nnz==0 and np.all(a.data==1)
    totals=np.asarray(x.sum(1)).ravel();inv=np.zeros_like(totals)
    np.divide(1.,totals,out=inv,where=totals!=0)
    features=(sp.diags(inv)@x).astype(np.float32).tocsr()
    a=a+sp.eye(2708,dtype=np.float32,format='csr')
    degree=np.asarray(a.sum(1)).ravel();d=sp.diags(np.power(degree,-.5))
    norm=(d@a@d).tocsr()
    train=np.zeros(2708,dtype=bool);val=train.copy();test=train.copy()
    train[:140]=True;val[140:640]=True;test[index]=True
    assert not np.any(train&val|train&test|val&test)
    assert [int(z.sum()) for z in [train,val,test]]==[140,500,1000]
    assert np.all(y[train].sum(0)==20)
    assert np.allclose(np.asarray(features.sum(1)).ravel(),1.,atol=1e-6)
    assert norm.nnz==13264 and np.count_nonzero(norm.diagonal())==2708
    diff=norm-norm.T;assert not diff.nnz or abs(diff.data).max()<1e-6
    assert np.all(np.isfinite(features.data)) and np.all(np.isfinite(norm.data))
    out.mkdir(parents=True,exist_ok=False)
    sp.save_npz(out/'features_norm.npz',features);sp.save_npz(out/'adj_norm.npz',norm)
    np.savez(out/'labels_and_masks.npz',labels=y.argmax(1).astype(np.int64),node_ids=np.arange(2708),train_mask=train,val_mask=val,test_mask=test)
    record={'reference_commit':COMMIT,'raw_sha256':expected,'processed_sha256':{n:digest(out/n) for n in ['features_norm.npz','adj_norm.npz','labels_and_masks.npz']},'statistics':{'nodes':2708,'features':1433,'classes':7,'undirected_edges':5278,'adjacency_nonzeros':13264,'split':[140,500,1000]},'passed':True}
    (out/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Cora preparation passed:',out)
if __name__=='__main__':main()
