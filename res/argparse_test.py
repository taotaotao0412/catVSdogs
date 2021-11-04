import argparse

parser = argparse.ArgumentParser(description='arg test')
parser.add_argument('--path', type=str, help='set dataset_path', required=True)
parser.add_argument('-v', type=str, help='version of project')

args = parser.parse_args()

print('path:', args.path)
print('version:', args.v)
