import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser(description='sample inputs to target num')
parser.add_argument('-i', '--input-file', type=str, help='the input sample files, comma split for multiple file')
parser.add_argument('-o', '--output', type=str, help='the output file to store the sample data')
parser.add_argument('-n', '--nums', type=int, help='The output sample nums')
parser.add_argument('-m', '--method', type=str, default='stratify', help="The sample method, default='stratify'")
parser.add_argument('-k', '--keys', type=str, default=None, help='The group by keys when doing stratify sample, comma splited input for multiple key')
args=parser.parse_args()


def sample_data(in_df, nums, method, keys=None):
    if method == 'stratify':
        if keys is None:
            print('"keys" should be specific when doing stratify sample')
        else:
            in_df['__group_by_column__'] = in_df[keys].astype(str).apply('_'.join, axis=1)
            list_of_sampled_groups = []
            total_len = len(in_df)
            for _, group in in_df.groupby('__group_by_column__'):
                group_len = len(group) * nums
                # here we should ceil the num in case the out sampled n less than num
                n_sample = int(group_len / total_len) if group_len % total_len == 0 else group_len // total_len + 1
                sampled_group = group.sample(n_sample)
                list_of_sampled_groups.append(sampled_group)
            sampled_data = pd.concat(list_of_sampled_groups).reset_index(drop=True)
            sampled_data.drop('__group_by_column__', axis=1, inplace=True)
            return sampled_data.sample(n=nums)
    else:
        print('Unsupported sample method')
        return None


if __name__ == '__main__':
    input_file_names = args.input_file.strip().split(',')
    input_dfs = []
    for f_name in input_file_names:
        df_part = pd.read_json(f_name, lines=True)
        input_dfs.append(df_part)
    df_input = pd.concat(input_dfs)
    group_keys = None
    if args.keys is not None:
        group_keys = args.keys.strip().split(',')
    out_datas = sample_data(df_input, args.nums, args.method, group_keys)
    out_datas.to_json(args.output, orient='records', indent=2, force_ascii=False)