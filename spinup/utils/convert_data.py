import os
import json
import pandas as pd

def convert_to_json(in_fname, out_fname=None):
    if out_fname is None:
        out_fname = in_fname[:in_fname.rfind('.')] + '.json'
    
    if out_fname[-5:] != '.json':
        return 'output file is not json'

    df = pd.read_table(in_fname)
    json_data = df.to_json(out_fname, orient='records', indent=4)
    
    return out_fname

def analysis(in_fname):
    df = pd.read_table(in_fname)
    print("mean:", df.mean())
    print("max:", df.max())
    print("min:", df.min())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infname', type=str, default=None)
    parser.add_argument('--outfname', type=str, default=None)
    args = parser.parse_args()

    convert_to_json(args.infname, args.outfname)
    # analysis(args.infname)