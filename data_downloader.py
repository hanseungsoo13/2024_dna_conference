from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import itertools

def process_data(data):
    url = data['url']
    boxes = data['ref_exps']
    caption = data['caption']
    height, width = data['height'], data['width']
    results = []

    for chunk in boxes:
        x_min = chunk[2] * width
        y_min = chunk[3] * height
        x_max = chunk[4] * width
        y_max = chunk[5] * height
        box = (x_min, y_min, x_max, y_max)
        noun = 'a photo of a ' + caption[int(chunk[0]):int(chunk[1])]
        results.append([url, box, noun])
    
    return results

ds = load_dataset("zzliang/GRIT")
all_data = pd.DataFrame(columns=['url', 'boxes', 'caption'])

with Pool() as pool:
    results = list(tqdm(pool.imap(process_data, ds['train']), total=len(ds['train'])))

# 결과를 10개의 구간으로 나눠서 데이터프레임 생성
num_chunks = 10
chunk_size = len(results) // num_chunks

for i in tqdm(range(num_chunks)):
    chunk_results = results[i * chunk_size:(i + 1) * chunk_size]
    chunk_results = list(itertools.chain.from_iterable(chunk_results))
    chunk_data = pd.DataFrame(chunk_results, columns=['url', 'boxes', 'caption'])
    chunk_data.to_csv(f'./cc/GRIT_data_chunk_{i + 1}.csv', sep='|', index=False)

# 마지막 남은 데이터 처리
remaining_results = results[num_chunks * chunk_size:]
if remaining_results:
    remaining_results = list(itertools.chain.from_iterable(remaining_results))
    chunk_data = pd.DataFrame(remaining_results, columns=['url', 'boxes', 'caption'])
    chunk_data.to_csv(f'./cc/GRIT_data_chunk_{num_chunks + 1}.csv', sep='|', index=False)

import gc

print('Start clearing memory...')
gc.collect()  # 메모리 정리

print('Start loading chunked data...')

# 모든 chunk 파일을 불러와서 하나의 데이터프레임으로 합치기
chunked_data = []
for i in range(1, num_chunks + 2):  # num_chunks + 1은 마지막 남은 데이터 처리
    chunk_data = pd.read_csv(f'./cc/GRIT_data_chunk_{i}.csv', sep='|')
    chunked_data.append(chunk_data)

all_data = pd.concat(chunked_data, ignore_index=True)

print('Start splitting data...')

train_data = all_data.sample(frac=0.7, random_state=42)
remaining_data = all_data.drop(train_data.index)
valid_data = remaining_data.sample(frac=0.2857, random_state=42)  # 0.2 / (0.2 + 0.1)
test_data = remaining_data.drop(valid_data.index)

print('Start saving data...')

train_data.to_csv('./cc/GRIT_train_data.csv', sep='|', index=False)
valid_data.to_csv('./cc/GRIT_valid_data.csv', sep='|', index=False)
test_data.to_csv('./cc/GRIT_test_data.csv', sep='|', index=False)