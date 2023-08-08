import pandas as pd
import os
import numpy as np
import argparse
import requests
import concurrent.futures
# from mpi4py import MPI
import torch
import warnings 

# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()
# SIZE = COMM.Get_size()

# ------------------------------------------------------------------------------
# distributed related
def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def dist_init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def request_save(url, save_fp):
    img_data = requests.get(url, timeout=5).content
    with open(save_fp, 'wb') as handler:
        handler.write(img_data)


def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    if RANK == 0:
        if not os.path.exists(os.path.join(video_dir, 'videos')):
            os.makedirs(os.path.join(video_dir, 'videos'))
    
    COMM.barrier()

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace('.csv', f'_{args.partitions}')

    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, subdf in enumerate(df_split):
            subdf.to_csv(os.path.join(partition_dir, f'{idx}.csv'), index=False)

    relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(os.path.join(args.data_dir, 'relevant_videos_exists.txt'), names=['fn'])
    else:
        exists = []

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    # data_dir/results_csvsplit/results_0.csv
    # data_dir/results_csvsplit/results_1.csv
    # ...
    # data_dir/results_csvsplit/results_N.csv


    df = pd.read_csv(os.path.join(partition_dir, f'{args.part}.csv'))

    df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])),
                            axis=1)

    df['rel_fn'] = df['rel_fn'] + '.mp4'

    df = df[~df['rel_fn'].isin(exists)]

    # remove nan
    df.dropna(subset=['page_dir'], inplace=True)

    playlists_to_dl = np.sort(df['page_dir'].unique())

    for page_dir in playlists_to_dl:
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir]
        if len(pdf) > 0:
            if not os.path.exists(vid_dir_t):
                os.makedirs(vid_dir_t)

            urls_todo = []
            save_fps = []

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
                if not os.path.isfile(video_fp):
                    urls_todo.append(row['contentUrl'])
                    save_fps.append(video_fp)

            print(f'Spawning {len(urls_todo)} jobs for page {page_dir}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {executor.submit(request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}
            # request_save(urls_todo[0], save_fps[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=4,
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, required=True,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path', type=str, default='results_2M_train.csv',
                        help='Path to csv data to download')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    dist_init()
    RANK = get_rank()
    SIZE = get_world_size()

    if SIZE > 1:
        warnings.warn("Overriding --part with MPI rank number")
        args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)
