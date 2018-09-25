import os
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='training of the baseline...')
    parser.add_argument('--gpus', type=str, default='',
                        help='gpu device\'s ID need to be used')
    args = parser.parse_args()
    return args


args = parse_arg()
print(args.gpus)
print(len(args.gpus))
print(args.gpus.split(','))
print(len(args.gpus.split(',')))
