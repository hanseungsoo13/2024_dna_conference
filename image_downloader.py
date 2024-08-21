import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os.path as osp
from multiprocessing import Pool, cpu_count


def check_image(args):
    url, caption, path = args
    save_path = None

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data)

            save_path = osp.join(path, f"{caption}.jpg")
            cnt = 0
            while osp.exists(save_path):
                cnt += 1

            if cnt != 0:
                save_path = osp.join(path, f"{caption}_{cnt}.jpg")
            
            image.save(save_path)

            return caption, save_path, url
        else:
            return caption, save_path, url
        
    except:
        return caption, save_path, url
    

def main(csv_path, save_path):
    data = pd.read_csv(csv_path, sep='|')
    new_data = pd.DataFrame(columns=['caption', 'img_path'])

    # Prepare arguments for multiprocessing
    args_list = [(row['img_link'], row['caption'], save_path) for _, row in data.iterrows()]

    # Set up multiprocessing pool
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_image, args_list), total=len(args_list)))

    # Process results
    for caption, img_path, url in results:
        if img_path is not None:
            new_data.loc[len(new_data)] = [caption, img_path]
    
    new_data.to_csv('./cc/val_data.csv', sep='|', index=False)


if __name__ == '__main__':
    csv_path = './cc/Validation_GCC-1.1.0-Validation_output.csv'
    save_path = './cc_data/val/'

    main(csv_path, save_path)