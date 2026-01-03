import sys
import argparse
import pandas as pd
from utils.quick_start import quick_start, inference_quick_start

def get_parser():
    description = 'Training model recommendation system'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, required=True, help='Name of model')
    parser.add_argument('--path_dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--config_dir', type=str, required=True, help='Folder of configs')
    parser.add_argument('--vision_feature_file', type=str, default=None, help='Path to Feature Images')
    parser.add_argument('--text_feature_file', type=str, default=None, help='Path to Features Description Books')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained model')
    parser.add_argument('--do_train', action='store_true', help='Whether to perform training')
    return parser

def main(args):
    config_dict = {
        'config_dir': args.config_dir,
        'vision_feature_file': args.vision_feature_file,
        'text_feature_file': args.text_feature_file
    }
    dataset = pd.read_csv(args.path_dataset)
    
    if args.do_train:
        quick_start(model=args.model, dataset=dataset, config_dict=config_dict, saved=args.save_model)
    else:
        inference_quick_start(model=args.model, dataset=dataset, config_dict=config_dict)

if __name__ == '__main__':
    args = get_parser().parse_args(sys.argv[1:])
    main(args)