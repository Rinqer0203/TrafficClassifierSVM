'''
トラフィックデータからLABELの値に基づいてランダムサンプリングを行うスクリプト
'''
import os
import random
from collections import defaultdict
from traffic_attributes import TrafficAttr

SAMPLED_SIZE = 10000


def get_sampled_data(path: str) -> str:
    '''
    各LABELのサンプルを取得し、統合して保存する
    '''
    file_paths = []
    if os.path.isdir(path):
        file_paths = [f'{path}/{filename}' for filename in os.listdir(path)
                      if os.path.isfile(f'{path}/{filename}')]
    else:
        file_paths = [path]

    # サンプリングするデータのファイル名を表示
    print('Sampling data from...')
    for path in file_paths:
        print(os.path.basename(path), end=', ')
    print()

    # 各LABELのサンプルを取得
    all_samples = extract_samples(file_paths, SAMPLED_SIZE)

    # 保存
    base_filename = os.path.basename(file_paths[0]).split('.')[0][:-2]
    out_path = f'{base_filename}_sampled.txt'
    with open(out_path, 'w') as f:
        # 一行目はヘッダーを書き込む
        header = TrafficAttr.get_attribute_name_list()
        f.write("\t".join(header) + "\n")
        f.write("\n".join(all_samples))
    print(f"Saved {len(all_samples)} samples to {out_path}.")

    return out_path


def extract_samples(file_paths, sample_size: int):
    """各LABELの値に基づいてランダムサンプリングを行う"""
    label_samples = defaultdict(list)
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                label = line.split('\t')[TrafficAttr.LABEL]
                if label == '-2':
                    label = '-1'  # -2を-1に統合
                    # lineのlabelを-1に変更
                    line = line.split('\t')
                    line[TrafficAttr.LABEL] = '-1'
                    line = '\t'.join(line)
                label_samples[label].append(line.strip())

    # 各LABELのサンプルを取得し、統合してシャッフル
    all_samples = []
    for label in ['1', '-1']:
        samples = label_samples[label]
        if len(samples) > sample_size:
            samples = random.sample(samples, sample_size)
        all_samples.extend(samples)

    random.shuffle(all_samples)

    return all_samples


if __name__ == '__main__':
    print('サンプリングするデータのファイルパスを入力してください')
    get_sampled_data(input())
