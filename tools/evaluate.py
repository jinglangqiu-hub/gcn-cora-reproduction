"""Evaluate the selected last.pt; write predictions to a new directory."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import torch
import numpy as np
from src.cora_data import load_cora_tensors
from src.gcn_model import GCN
from src.mlp_model import MLP
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed/cora'))
    a=p.parse_args();cfg=json.loads((a.run_dir/'config.json').read_text())
    result=json.loads((a.run_dir/'result.json').read_text())
    if result['selected_checkpoint']!='last.pt':raise ValueError('Only the preselected last.pt is supported')
    weight=a.run_dir/'last.pt';digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest==result['checkpoint_sha256']['last.pt']
    verified=json.loads((a.data_dir/'verification.json').read_text())
    for name,wanted in verified['processed_sha256'].items():assert hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest()==wanted
    torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    x,adj,y,tr,va,te=load_cora_tensors(a.data_dir)
    saved=torch.load(weight,map_location='cpu',weights_only=True)
    model=(MLP if cfg['model']=='mlp' else GCN)(1433,16,7,cfg['dropout'])
    model.load_state_dict(saved['model_state_dict']);model.eval()
    assert saved['epoch']==result['selected_epoch']
    with torch.no_grad():
        logits=model(x) if cfg['model']=='mlp' else model(x,adj)
        assert (logits[va].argmax(1)==y[va]).float().mean().item()==result['last_validation_accuracy']
        probs=logits[te].softmax(1);prediction=probs.argmax(1);truth=y[te]
    node_ids=np.flatnonzero(te.numpy());correct=int((prediction==truth).sum())
    a.output_dir.mkdir(parents=True,exist_ok=False)
    fields=['node_id','true_class','predicted_class']+[f'prob_class_{i}' for i in range(7)]
    with (a.output_dir/'predictions.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,node in enumerate(node_ids):
            row=dict(node_id=int(node),true_class=int(truth[i]),predicted_class=int(prediction[i]))
            row.update({f'prob_class_{c}':float(probs[i,c]) for c in range(7)});w.writerow(row)
    score={'model':cfg['model'],'seed':cfg['seed'],'epoch':saved['epoch'],'correct':correct,'test_nodes':1000,'test_accuracy':correct/1000,'checkpoint_sha256':digest,'optimizer_updates':0,'command':sys.orig_argv}
    (a.output_dir/'test_result.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    print(json.dumps(score,indent=2))
if __name__=='__main__':main()
