import os 
import argparse 

def Run(args):

    if args.dataset == "PR":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset == "PA":
        path = args.dataset_path + "/paper100M/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 11105995
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset == "CO":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset == "UKS":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset == "UKL":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset == "CL":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    else:
        print("invalid dataset path")
        return

    with open("meta_config","w") as file:
        file.write("{} {} {} {} {} {} {} {} {} {} {}".format(path, args.train_batch_size, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, test_set_num, args.cache_memory, args.epoch, 1-args.usenvlink))

    gpu_number = args.gpu_number
    if args.usenvlink == 1:
        if gpu_number >= 2:
            cache_agg_mode = 1
        else:
            cache_agg_mode = 0
    else:
        cache_agg_mode = 0
    os.system("./src/legion {} {}".format(gpu_number, cache_agg_mode))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Legion Server.")
    argparser.add_argument('--dataset_path', type=str, default="/home/atc-artifacts-user/datasets")
    argparser.add_argument('--dataset', type=str, default="PA")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=1)
    argparser.add_argument('--epoch', type=int, default=10)
    argparser.add_argument('--cache_memory', type=int, default=38000000000)
    argparser.add_argument('--usenvlink', type=int, default=1)
    args = argparser.parse_args()
    Run(args)
