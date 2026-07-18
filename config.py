import argparse
import os

def parse_args():
    parser=argparse.ArgumentParser(description="arguments")
    #hyperparameters
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    parser.add_argument('--max_epoch',type=int,default=50)
    #paths  
    parser.add_argument('--cifar_path',type=str,default='./dataset/cifar-10')
    parser.add_argument('--checkpoint_path',type=str,default='./checkpoint/cifar-10')
    parser.add_argument('--preprocessed_data_path',type=str,default='./preprocessed_data/cifar-10')
    #others
    parser.add_argument('--model_name',type=str,default='test_CNN')

    args=parser.parse_args()

    os.makedirs(args.cifar_path,exist_ok=True)
    os.makedirs(args.checkpoint_path,exist_ok=True)
    os.makedirs(args.preprocessed_data_path,exist_ok=True)

    return args

def show_args(args):
    print('==================================\n')
    for key,value in vars(args).items():
        print(f'{key}:{value}\n')
    print('==================================\n')

def save_args(path,args):
    with open(path,'w') as f:
        for key,value in vars(args).items():
            f.write(f'{key}:{value}\n')

def main():
    args=parse_args()
    show_args(args)
    path='./args'
    os.makedirs(path,exist_ok=True)
    name='test.txt'
    joined_path=os.path.join(path,name)
    save_args(path=joined_path,args=args)

if __name__=='__main__':
    main()
