from config.config import get_config
from core.executor import *
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Exp Args')

    parser.add_argument("--mode", choices=['train', 'eval', 'predict'],
                      help='{train, eval, predict}',
                      type=str, required=True)
    
    parser.add_argument("--evaltype", choices=['last', 'best'],
                      help='{last, best}',
                      type=str, nargs='?', const=1, default='last')
    parser.add_argument("--predicttype", choices=['last', 'best'],
                      help='{last, best}',
                      type=str, nargs='?', const=1, default='best')
    
    parser.add_argument("--config-file", type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    config = get_config(args.config_file)

    exec = getattr(sys.modules[__name__], config.EXECUTOR)(config, args.mode, args.evaltype, args.predicttype)
    exec.run()