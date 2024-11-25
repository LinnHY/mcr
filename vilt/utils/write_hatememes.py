import json, jsonlines
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word

def make_arrow(root, dataset_root, single_plot=False):
    split_sets = ['train', 'test']
    fin_name = '/data/lhy/missing_aware_prompts/datasets/Hatefull_Memes/data'
    
    for split in split_sets:
        with open(fin_name+'/caption_'+split+'.json', 'r') as fp:
            content = json.load(fp)
        data_list = []
        with jsonlines.open(os.path.join(root,f'data/{split}.jsonl'), 'r') as rfd:
            for data in tqdm(rfd):
                image_path = os.path.join(root, 'data', data['img'])
                
                with open(image_path, "rb") as fp:
                    binary = fp.read()       
                    
                text = [data['text']]
                label = data['label']
                id=str(data['id'])
                if len(id)==4:
                    id='0'+id
                caption=content[id]
                #text_aug = text_aug_dir['{}.png'.format(data['id'])]

                data = (binary, text, caption, label, split)
                data_list.append(data)                
                            

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "caption",
                "label",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/hatememes_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        