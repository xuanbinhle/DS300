import sys
import argparse
import pandas as pd
from utils.quick_start import quick_start

def get_parser():
    description = 'Training model recommendation system'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, required=True, help='Name of model')
    parser.add_argument('--path_dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--config_dir', type=str, required=True, help='Folder of configs')
    return parser

def main(args):
    config_dict = {
        'config_dir': args.config_dir
    }
    dataset = pd.read_csv(args.path_dataset)
    quick_start(model=args.model, dataset=dataset, config_dict=config_dict)

if __name__ == '__main__':
    args = get_parser().parse_args(sys.argv[1:])
    main(args)