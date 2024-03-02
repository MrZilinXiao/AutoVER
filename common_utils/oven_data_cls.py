from torch.utils.data import Dataset
import torch
import transformers
import os
from typing import List, Dict, Optional, Sequence, Tuple
import json
from PIL import Image
from LLaVA.llava.constants import (IGNORE_INDEX, 
                                   DEFAULT_IMAGE_TOKEN, DEFAULT_RET_TOKEN, 
                                   DEFAULT_ENT_CLS_TOKEN, DEFAULT_ENT_CLS_TOKEN_INDEX)
from LLaVA.llava.mm_utils import tokenizer_image_token
from dataclasses import dataclass


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def append_space_if_not_ends_with_space(s):
    if not s.endswith(' '):
        s += ' '
    return s
    
def preprocess_oven_with_cls(
    data,
    tokenizer,
    qid_to_cls_dict, 
    sep_token = '\n',   # common practice in llava training
    eos_token = None,   # should be None as we need it to end
    # training or not in cls mode does not matter
):
    if eos_token is None: 
        eos_token = tokenizer.eos_token
    
    question_text = DEFAULT_IMAGE_TOKEN + sep_token + data['question']
    sample_text = question_text + DEFAULT_ENT_CLS_TOKEN

    input_ids = tokenizer_image_token(sample_text, tokenizer, return_tensors='pt').unsqueeze(0)  # insert IMAGE_TOKEN_INDEX into input_ids
    targets = IGNORE_INDEX * torch.ones_like(input_ids)  # 1, L
    # update the last pos of targets to be qid
    cls_idx = qid_to_cls_dict[data['entity_id']]
    targets[:, -1] = cls_idx

    return dict(input_ids=input_ids, labels=targets)


class OvenJsonlDatasetCLS(Dataset):
    # training jsonl format: 
    # {
    #     "data_id": "oven_entity_train_00000000",
    #     "image_id": "oven_00000000",
    #     "question": "what is the model of this aircraft?",
    #     "entity_id": "Q1141409",
    #     "entity_text": "Dassault Falcon 900",
    #     "data_split": "entity_train"
    # }
    POSSIBLE_IMG_EXT = ('.jpg', '.JPEG')  # checked
    EVAL_NEEDED_KEYS = ('question', 'data_id', 'entity_id', 'entity_text', 'data_split')

    # allow missing as test split does not give answer
    def __init__(self, jsonl_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args, 
                 img_shards_path: str, is_training=True, 
                 # below are passed from init_dataset_kwargs
                 wiki_summary_path: Optional[str]=None, is_summary=False,   # if with wiki_full_path, we attach description of img
                 max_summary_tokens=128, 
                 # below are for retrieval: note that retrieval also requires wiki_summary_path for description
                 retrieval: bool = False, entity_image_processor=None, entity_tokenizer=None, 
                 entity_img_path: str = None, 
                 # fixed on 11/07: add qid_to_entity_text, s.t. entity surface form gets unified.
                 # previous checkpoints will be slightly affected.
                qid_to_entity_path: Optional[Dict[str, str]]=None,  # if not None, we use this to replace entity_text
                # otherwise we keep compability with checkpoints before 11/07
                qid_to_cls_path: Optional[Dict[str, int]]=None,  # if not None, we use this to replace entity_text
                 ) -> None:
        super().__init__()
        self.img_shards_path = img_shards_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.is_training = is_training
        with open(jsonl_path, "r") as f:
            self.data: List[dict] = [json.loads(line) for line in f]

        self.qid_to_entity_dict = None
        if qid_to_entity_path is not None:
            with open(qid_to_entity_path, 'r') as f:
                self.qid_to_entity_dict = json.load(f)
        else:
            print("qid_to_entity_path is None, we use entity_text in the dataset; this is not recommended.")

        with open(qid_to_cls_path, 'r') as f:
            self.qid_to_cls_dict = json.load(f)

    def find_img_path(self, image_id: str):
        shard_id = image_id.split('_')[1][:2]
        for ext in self.POSSIBLE_IMG_EXT:
            img_path = os.path.join(self.img_shards_path, f"{shard_id}", f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        raise RuntimeError(f"Image {image_id} not found in {img_path}")
    
    def find_entity_img_path(self, entity_id: str):
        shard_id = entity_id[:4]  # Q1, Q11, Q111 all covers
        for ext in self.POSSIBLE_IMG_EXT:
            img_path = os.path.join(self.entity_img_path, f"{shard_id}", f"{entity_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        # raise RuntimeError(f"Image {entity_id} not found in {img_path}")
        return None   # missing img is possible; we should fill a black image
    
    def __len__(self):
        return len(self.data)
    
    @property
    def lengths(self):  # for length grouped training; however this does not count tokens but words?
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'image_id' in sample else 0  # does img_tokens take that much?
            # length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
            length_list.append(sum(len((sample['question'] + sample['entity_text']).split())) + img_tokens)
        return length_list
    
    @property
    def entity_labels(self):  # get all entity labels in entity QID, so that we can build a balanced sampler
        label_list = []
        for sample in self.data:
            label_list.append(int(sample['entity_id'][1:]))
        return label_list
    
    def __getitem__(self, index):
        # keys needed: 
        # 1. input_ids, labels
        # 2. image
        # preprocess: insert DEFAULT_IMAGE_TOKEN before every question.
        data = self.data[index]
        entity_id = data['entity_id']
        if self.qid_to_entity_dict is not None:
            data['entity_text'] = self.qid_to_entity_dict[entity_id]
        
        image_id = data['image_id']
        image_path = self.find_img_path(image_id)
        processor = self.data_args.image_processor
        # oven data are always accompanied with an image
        image = Image.open(image_path).convert('RGB')
        if self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # preprocess gets unified in our impl
        # data_dict = preprocess_oven(data, self.tokenizer, self.data_args, 
        data_dict = preprocess_oven_with_cls(
            data, self.tokenizer,
            self.qid_to_cls_dict
        )
        # remove the redundent batch dim
        data_dict = {k: v.squeeze(0) for k, v in data_dict.items()}
        data_dict['image'] = image
        # not in is_training will drop entity_text
        # you should prepare external labels for eval 
        # (entity / query set, seen / unseen etc.)
        # collect wiki image

        if not self.is_training:
            for k in self.EVAL_NEEDED_KEYS:
                try:
                    data_dict[k] = data[k]
                except KeyError:
                    pass   # missing due to test split
        return data_dict  # wait for data_colltor
    
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # let data collator skipps all evaluation keys, which we use for evaluation
        if any(k in instances[0] for k in OvenJsonlDatasetCLS.EVAL_NEEDED_KEYS):
            for k in OvenJsonlDatasetCLS.EVAL_NEEDED_KEYS:
                if k in instances[0]:
                    # grouped by batch for eval
                    batch[k] = [instance[k] for instance in instances]

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)  # ensured by preprocessor
            else:
                batch['images'] = images
        
        return batch
    

def build_oven_data_modules_cls(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args,
    is_training = True, 
):
    assert DEFAULT_ENT_CLS_TOKEN in tokenizer.get_vocab(), f"CLS token {DEFAULT_ENT_CLS_TOKEN} not in vocab!"
    train_dataset = OvenJsonlDatasetCLS(data_args.train_jsonl_path, tokenizer, data_args, 
                                     data_args.train_img_shards_path, is_training=is_training, 
                                     wiki_summary_path=data_args.qid_to_summary_path,
                                     is_summary=data_args.use_summary,
                                     max_summary_tokens=data_args.max_summary_tokens,
                                     retrieval=data_args.retrieval,
                                     entity_image_processor=data_args.entity_image_processor,
                                     entity_tokenizer=data_args.entity_tokenizer, 
                                     entity_img_path=data_args.entity_img_path,
                                     qid_to_entity_path=data_args.train_qid_to_entity_path,
                                     qid_to_cls_path=data_args.qid_to_cls_path,)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)  

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


if __name__ == '__main__':
    # see `cls_playground.ipynb` for unit test
    pass