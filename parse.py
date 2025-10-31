from neural_network import *


def parse_method(args, c, d, device):
    if args.method == 'ours':
        model = HybridBlock(d, args.hidden_channels, c, trans_num_layers=args.ours_layers, alpha=args.alpha,
                            trans_dropout=args.ours_dropout, trans_num_heads=args.num_heads, trans_use_bn=args.use_bn,
                            trans_use_residual=args.ours_use_residual, trans_use_weight=args.ours_use_weight,
                            trans_use_act=args.ours_use_act,
                            gnn_num_layers=args.num_layers, gnn_dropout=args.dropout, gnn_use_bn=args.use_bn,
                            gnn_use_residual=args.use_residual, gnn_use_weight=args.use_weight,
                            gnn_use_init=args.use_init,
                            gnn_use_act=args.use_act,
                            use_graph=args.use_graph, graph_weight=args.graph_weight, aggregate=args.aggregate,
                            trans_method=args.trans_method).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='strength of L2 regularization on GNN weights')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=str, default='cuda:0', help='use pseudo-label learning')
    parser.add_argument('--val_best', type=str, default='loss')

    # model
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers for deep methods')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--trans_method', type=str, default='graph_add_trans')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight')
    parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--ours_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--ours_layers', type=int, default=1, help='gnn layer.')
    parser.add_argument('--ours_dropout', type=float, default=0.5, help='gnn dropout.')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')

    # pseudo-label learning
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='sample ratio of true-labeled nodes')
    parser.add_argument('--threshold_ratio', type=float, default=0.5, help='threshold ratio of community in pesudo-labeled nodes')
    parser.add_argument('--real_label_threshold', type=float, default=0.2,
                        help='threshold for the proportion of true-labeled nodes within each weak clique')
    parser.add_argument('--first_true_label_weight', type=int, default=1, help='true-label weight')
    parser.add_argument('--first_pseudo_label_weight', type=int, default=1, help='pseudo-label weight')
    parser.add_argument('--second_true_label_weight', type=int, default=1, help='true-label weight')
    parser.add_argument('--second_pseudo_label_weight', type=int, default=1, help='pseudo-label weight')
    parser.add_argument('--use_pseudo', action='store_true', help='use pseudo-label learning')
    parser.add_argument('--community_confident_threshold', type=int, default=0.5, help='filter community affiliation after first round training')


def change_input_args(args, dataset_path):
    args.use_residual = True
    args.use_weight = True
    args.use_bn = True
    args.use_init = True
    args.use_act = True
    args.ours_use_residual = True
    args.ours_use_weight = True
    args.ours_use_bn = True
    args.use_graph = True
    args.lr = 1e-3
    args.weight_decay = 1e-4
    args.device = 'cuda:1'
    args.sample_ratio = 0.10

    file_name = os.path.basename(dataset_path)
    file_stem = file_name.split('.')[0]

    if file_stem == 'fb_0':
        args.threshold_ratio  = 0.50
        args.real_label_threshold = 0.05
        args.max_epochs = 1000
        args.true_label_weight = 3
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'fb_107':
        args.threshold_ratio  = 0.50
        args.real_label_threshold = 0.05
        args.max_epochs = 500
        args.true_label_weight = 1
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'fb_1684':
        args.threshold_ratio  = 0.5
        args.real_label_threshold = 0.05
        args.max_epochs = 500
        args.true_label_weight = 3
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'loss'
        args.seed = 3

    if file_stem == 'fb_1912':
        args.threshold_ratio  = 0.8
        args.real_label_threshold = 0.05
        args.max_epochs = 500
        args.true_label_weight = 3
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'mag_eng':
        args.threshold_ratio  = 0.8
        args.real_label_threshold = 0.20
        args.max_epochs = 150
        args.true_label_weight = 5
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'mag_cs':
        args.threshold_ratio  = 0.8
        args.real_label_threshold = 0.20
        args.max_epochs = 150
        args.true_label_weight = 5
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'mag_chem':
        args.threshold_ratio  = 0.8
        args.real_label_threshold = 0.20
        args.max_epochs = 150
        args.true_label_weight = 5
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3

    if file_stem == 'mag_med':
        args.threshold_ratio  = 0.8
        args.real_label_threshold = 0.2
        args.max_epochs = 150
        args.true_label_weight = 5
        args.pseudo_label_weight = 1
        args.use_pseudo = True
        args.community_confident_threshold = 0.5
        args.val_best = 'onmi'
        args.seed = 3