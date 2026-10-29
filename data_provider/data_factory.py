from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_EpiGNN,
    Dataset_ETT_hour,
    Dataset_M4,
    Dataset_Solar,
    Dataset_TSF,
    Dataset_TSF_ICL,
)
from data_provider.graph_utils import (
    adjacency_from_payload,
    build_distance_weight_matrix,
    build_gravity_weight_matrix,
    get_population_vector,
    load_region_dataframe,
    normalize_adjacency_matrix,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL,
    'epidataset': Dataset_EpiGNN,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        batch_size = args.batch_size

    dataset_kwargs = {
        'root_path': args.root_path,
        'data_path': args.data_path,
        'flag': flag,
    }
    if args.data == 'epidataset':
        dataset_kwargs.update({
            'graph_mode': getattr(args, 'mobility_graph_mode', 'empirical'),
            'gravity_alpha': getattr(args, 'gravity_alpha', 1.0),
            'gravity_beta': getattr(args, 'gravity_beta', 1.0),
            'gravity_gamma': getattr(args, 'gravity_gamma', 2.0),
        })

    if flag in ['train', 'val']:
        dataset_kwargs['size'] = [args.seq_len, args.label_len, args.token_len]
    else:
        dataset_kwargs['size'] = [args.test_seq_len, args.test_label_len, args.test_pred_len, args.token_len]

    data_set = Data(**dataset_kwargs)

    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))
    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
        )
    return data_set, data_loader, data_set.scaler


def get_adj(
    country,
    root_path=None,
    data_path=None,
    graph_mode='distance',
    gravity_alpha=1.0,
    gravity_beta=1.0,
    gravity_gamma=2.0,
):
    _, region_df, payload = load_region_dataframe(
        country=country,
        root_path=root_path,
        data_path=data_path,
    )

    if graph_mode == 'distance':
        adj = build_distance_weight_matrix(region_df)
    elif graph_mode == 'gravity':
        population = get_population_vector(region_df, payload=payload)
        adj = build_gravity_weight_matrix(
            region_df,
            population,
            alpha=gravity_alpha,
            beta=gravity_beta,
            gamma=gravity_gamma,
        )
    elif graph_mode == 'empirical':
        if payload is None:
            raise ValueError(
                f'Empirical graph mode requires a JSON payload with mobility data for `{country}`.'
            )
        adj = adjacency_from_payload(payload)
    else:
        raise ValueError(
            f'Unsupported graph_mode `{graph_mode}`. Choose from distance, gravity, empirical.'
        )

    return normalize_adjacency_matrix(adj, add_self_loops=True, scale_by_max=True)
