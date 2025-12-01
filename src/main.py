import sys
import argparse
from utils.quick_start import quick_start

def get_parser():
    description = 'Training model recommendation system'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, required=True, help='Name of model')
    parser.add_argument('--config_dir', type=str, required=True, help='Folder of configs')
    return parser

def main(args):
    config_dict = {
        'config_dir': args.config_dir
    }
    quick_start(model=args.model, config_dict=config_dict, save_model=True)

if __name__ == '__main__':
    args = get_parser().parse_args(sys.argv[1:])
    main(args)