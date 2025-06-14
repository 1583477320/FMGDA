from utils.options import args_parser

args = args_parser()
print(args.method)

args.method = 'fm'
print(args.method)