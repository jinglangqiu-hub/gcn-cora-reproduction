"""Run fresh GCN/MLP experiments under a new batch identity."""
import argparse,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for seed in range(10):
        for model in ['gcn','mlp']:
            run=a.output_dir/f'{model}_seed{seed:02d}'
            subprocess.run([sys.executable,'-B','-m','tools.train','--model',model,'--seed',str(seed),'--run-dir',str(run)],check=True)
            subprocess.run([sys.executable,'-B','-m','tools.evaluate','--run-dir',str(run),'--output-dir',str(a.output_dir/f'eval_{model}_seed{seed:02d}')],check=True)
if __name__=='__main__':main()
