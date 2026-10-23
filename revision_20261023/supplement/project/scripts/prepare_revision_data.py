"""Build versioned, matched forecasting samples from observed inputs only."""
import sys
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.revision.data import load_ddatshr, load_diadata, save_dataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['ddatshr','diadata','all'],default='all')
    parser.add_argument('--history',type=int,default=120);args=parser.parse_args()
    out=ROOT/'data/processed/revision_v2'/f'history_{args.history}'
    for name,loader in [('ddatshr',load_ddatshr),('diadata',load_diadata)]:
        if args.dataset not in ['all',name]:continue
        grid,audit=loader(ROOT/'data/raw'/name)
        result=save_dataset(grid,audit,out,name,args.history)
        print({k:result[k] for k in ('dataset','samples','patients','array_sha256')},flush=True)


if __name__=='__main__':main()
