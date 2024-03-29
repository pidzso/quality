import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',            type=int,   default=0,       help='random seed')
    parser.add_argument('--dataset',         type=str,   default='mnist', help="name of dataset")
    parser.add_argument('--model',           type=str,   default='mlp',   help='model name')
    parser.add_argument('--num_users',       type=int,   default=5,       help="number of users: K")
    parser.add_argument('--frac',            type=float, default=1,     help='the fraction of clients: C')
    parser.add_argument('--epochs',          type=int,   default=10,      help="number of rounds of training")

    parser.add_argument('--weight',          type=float, default=0.,      help='weight participants')
    parser.add_argument('--aggregate',       type=str,   default='avg',   help="type of aggregation")
    parser.add_argument('--robustness',      type=int,   default=0,       help="number of resisting attackers")
    parser.add_argument('--noise_type',      type=str,   default='no',    help='noise type')
    parser.add_argument('--noise_size',      type=float, default=0.,      help='noise parameter')

    parser.add_argument('--r_check',         type=int, default=2,         help='round to measure contribution')
    parser.add_argument('--anon_set',        type=int, default=1,         help='size of groups to be tested')
    parser.add_argument('--groupping',       type=str, default='',        help='define groups to be tested')

    parser.add_argument('--optimizer',       type=str,   default='sgd',   help="type of optimizer")
    parser.add_argument('--lr',              type=float, default=0.01,    help='learning rate')
    parser.add_argument('--do',              type=float, default=0.1,     help='drop out')
    parser.add_argument('--norm',            type=str,   default='None',  help="batch_norm, layer_norm, or None")
    parser.add_argument('--local_ep',        type=int,   default=1,       help="the number of local epochs: E")
    parser.add_argument('--local_bs',        type=int,   default=32,      help="local batch size: B")
    parser.add_argument('--momentum',        type=float, default=0.9,     help='SGD momentum')
    parser.add_argument('--kernel_num',      type=int, default=9,         help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes',    type=str, default='3,4,5',   help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels',    type=int, default=1,         help="number of channels of imgs")
    parser.add_argument('--num_filters',     type=int, default=32,        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool',        type=str, default='True',    help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_classes',     type=int, default=10,        help="number of classes")
    parser.add_argument('--stopping_rounds', type=int, default=10,        help='rounds of early stopping')
    parser.add_argument('--verbose',         type=int, default=1,         help='verbose')

    parser.add_argument('--gpu',             type=int, default=None,      help="To use cuda, set to a specific GPU ID. Default set to use CPU.")

    args = parser.parse_args()

    return args
