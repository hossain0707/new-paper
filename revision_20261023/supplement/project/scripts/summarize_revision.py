import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.revision.reporting import collect,summarize_predictions

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['ddatshr','diadata'])
    p.add_argument('--output',default='results/revision_v2');a=p.parse_args()
    f=collect(ROOT/'data/processed/revision_v2/history_120',ROOT/a.output,a.datasets)
    macro,comparisons=summarize_predictions(f,ROOT/a.output)
    print(macro[macro.seed.eq('mean')].to_string(index=False));print(comparisons.to_string(index=False))
