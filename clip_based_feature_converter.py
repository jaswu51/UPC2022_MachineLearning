import pandas as pd
import clip
from tqdm import tqdm
import os
import torch
from PIL import Image
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
print(os.getcwd())

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_black_picture(empty_img_path):
    import cv2 as cv
    import numpy as np
    def create_image():
        img = np.zeros([1200, 1750, 3], np.uint8)
        cv.imwrite(empty_img_path, img)

    create_image()

def multi_preprocess(img_file_name, preprocess):
    return preprocess(Image.open(img_file_name))

def feature_converter(article_file_name, batch_size):
    article_df = pd.read_csv(article_file_name, header=0, skiprows=range(1, int(args.start)), nrows=int(args.nrows), dtype={'article_id':str})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    output_path = "../data/image_text_feature_from_"+args.start+"_to_"+str(int(args.start)+int(args.nrows))+".csv"
    if (os.path.exists(output_path)):
        os.remove(output_path)
    # partial_work = partial(multi_preprocess, preprocess=preprocess)
    # p = Pool(processes=4)
    print_header = True
    line_index = int(args.start)
    for rows in tqdm(chunks(list(article_df.iterrows()), batch_size), total=article_df.shape[0]/batch_size):
        images = []
        texts = []
        feature_list = []
        for i in range(len(rows)):
            row = rows[i]
            article_id = str(dict(row[1])['article_id'])
            image_dir = "../data/images/"+article_id[:3]
            img_file_name = image_dir + "/"+article_id+".jpg"
            text_desc = dict(row[1])['detail_desc']
            if os.path.exists(image_dir) and os.path.exists(img_file_name):
                pass
            else:
                img_file_name = '../data/images/empty_black.jpg'
                print("empty image")
                if not os.path.exists(img_file_name):
                    create_black_picture(img_file_name)

            image = preprocess(Image.open(img_file_name))
            images.append(image)
            text = clip.tokenize([str(text_desc)[:77]]).squeeze(0)
            texts.append(text)

        image_features = model.encode_image(torch.stack(images).to(device))
        text_features = model.encode_text(torch.stack(texts).to(device))
        for j in range(image_features.size(dim=0)):
            image_features_list = [float(x) for x in image_features[j].tolist()]
            text_features_list = [float(x) for x in text_features[j].tolist()]
            feature_list.append(image_features_list+text_features_list)
        info_df = pd.DataFrame([pd.Series(r[1]) for r in rows])
        feature_df = pd.DataFrame(feature_list, columns=['image_'+str(j) for j in range(len(image_features_list))]+['detail_desc_'+str(j)
                                                            for j in range(len(text_features_list))])
        info_index = info_df.index
        info_df.reset_index(drop=True, inplace=True)
        feature_df.reset_index(drop=True, inplace=True)
        new_df = pd.concat([info_df, feature_df], axis=1)
        new_df = new_df.set_index(np.arange(line_index, line_index+len(rows)))
        new_df.to_csv(output_path, mode="a", header= print_header)
        line_index += int(batch_size)

        print_header = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nrows", "--nrows", help="Number of Row")
    parser.add_argument("-start", "--start", help="Start line")
    args = parser.parse_args()
    article_file_name = "../data/articles.csv"

    batch_size = 16
    feature_converter(article_file_name, batch_size)
