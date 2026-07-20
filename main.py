import torch
import config

def main():
    print('0')
    args=config.parse_args()
    config.show_args(args)

if __name__=='__main__':
    main()