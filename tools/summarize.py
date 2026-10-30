"""Independently check every fixed-split result, then summarize all ten pairs."""
import csv,json,statistics,argparse
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,default=Path('results'));a=p.parse_args()
    with (a.results_dir/'test_runs.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==20 and {(r['model'],int(r['seed'])) for r in rows}=={(m,s) for m in ['gcn','mlp'] for s in range(10)}
    summary=json.loads((a.results_dir/'test_summary.json').read_text())
    for m in ['gcn','mlp']:
        values=[float(r['test_accuracy']) for r in rows if r['model']==m]
        actual=summary['models'][m]
        assert abs(statistics.mean(values)-actual['test_mean'])<1e-14
        assert abs(statistics.stdev(values)-actual['test_sample_std_ddof1'])<1e-14
        print(f'{m}: {statistics.mean(values)*100:.2f}% +/- {statistics.stdev(values)*100:.2f} pp (sample std, n=10)')
    scores={(r['model'],int(r['seed'])):float(r['test_accuracy']) for r in rows}
    paired=[scores[('gcn',s)]-scores[('mlp',s)] for s in range(10)]
    assert min(paired)>0 and abs(statistics.mean(paired)-summary['paired_difference']['mean_gcn_minus_mlp'])<1e-14
    print('All 20 runs retained; paired mean gap:',round(statistics.mean(paired)*100,2),'pp')
if __name__=='__main__':main()
