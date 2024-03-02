from torch.utils.data import Dataset, ConcatDataset
import torch
import transformers
import os
from typing import List, Dict, Optional, Sequence, Tuple
import json
import copy
from PIL import Image
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
                                   DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
                                   DEFAULT_IM_END_TOKEN, DEFAULT_RET_TOKEN)
from LLaVA.llava.mm_utils import tokenizer_image_token
from dataclasses import dataclass
import spacy
from common_utils.split_utils import get_max_sentences


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
    
def prepare_template(tokenizer, question, sep_token, eos_token, 
                     entity_text=None,
                     summary=None, is_training=True, 
                     question_intro="Question: ", 
                     summary_intro="Possible Entity Description: ", 
                    #  answer_intro="Answer Entity: "
                     is_retrieval=False, 
                     is_summary=False, 
                     spacy_model=None, 
                     max_summary_tokens=128, 
                     ) -> Tuple[str, int]:
    enable_summary = summary is not None and is_training  # in eval, we don't give summary

    summary = get_max_sentences(spacy_model, summary, 
                            tokenizer_len_func=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), 
                            max_tokens=max_summary_tokens)
    enable_summary = enable_summary and (isinstance(summary, str) and len(summary) > 0)  # if summary is empty, we don't use it

    prompt_text = DEFAULT_IMAGE_TOKEN + sep_token
    if summary is not None:
        # train with summary
        # print("summary is not None branch...")
        prompt_text += question_intro + question + " " + summary_intro
    elif summary is None and is_summary:
        # val with summary; to save mem we don't load summary in val so summary will be None
        # print("summary is None and is_summary branch...")
        prompt_text += question_intro + question + " " + summary_intro
    else:
        # raw: is_summary is False, which is usually controlled by args 
        # prompt_text += question
        prompt_text += question
    prompt_text = prompt_text.strip()  # remove trailing space if any; VERY IMPORTANT!
    # we only give concat descrtiption in training; in eval we allow free generation
    no_modeling_index = len(tokenizer_image_token(prompt_text, tokenizer))
    sep_token_1 = sep_token_2 = sep_token
    if is_retrieval:  # the second sep gets updated to RET
        sep_token_2 = DEFAULT_RET_TOKEN

    if enable_summary:
        # train with a valid summary
        prompt_text += sep_token_1 + summary + sep_token_2 + entity_text + eos_token
    elif is_training:
        # training but no (valid) summary, we model entity_text
        prompt_text += sep_token_2 + entity_text + eos_token
    else:
        # eval does not need entity_text
        prompt_text += sep_token_2
    return prompt_text, no_modeling_index


def preprocess_oven_with_summary(
    data: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,  # only use `mm_use_im_start_end`
    sep_token = '\n',   # common practice in llava training
    eos_token = None,   # should be None as we need it to end
    is_training = True, 
    is_retrieval = False,  # <ret> token should always be placed after <query>/<desc>
    is_summary = False, 
    spacy_model = None, 
    max_summary_tokens = 128, 
) -> Dict[str, torch.Tensor]:
    if eos_token is None:  # though not typical in llava, we prefer eos to end the generation
        eos_token = tokenizer.eos_token

    prompt_text, src_len = prepare_template(
        tokenizer, data['question'], 
        sep_token, eos_token, 
        entity_text=data['entity_text'] if is_training else None, 
        summary=data['entity_summary'] if is_summary else None,
        is_training=is_training,
        is_retrieval=is_retrieval,
        is_summary=is_summary,
        spacy_model=spacy_model,
        max_summary_tokens=max_summary_tokens,
    )
    input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors='pt').unsqueeze(0)  # insert IMAGE_TOKEN_INDEX into input_ids
    targets = input_ids.clone()  # 1, L
    # targets[:, :src_len] = IGNORE_INDEX   # does not include the <ret> token / sep_token_2
    targets[:, :src_len] = IGNORE_INDEX   # include the <ret> token / sep_token_2

    return dict(input_ids=input_ids, labels=targets)

# just keep for compatibility; won't be used
# def preprocess_oven(
#     data: Dict[str, str],
#     tokenizer: transformers.PreTrainedTokenizer,
#     data_args,  # only use `mm_use_im_start_end`
#     sep_token = '\n',   # common practice in llava training
#     eos_token = None,   # should be None as we need it to end
#     is_training = True, 
#     is_summary = False,
# ) -> Dict:
#     if eos_token is None:
#         eos_token = tokenizer.eos_token

#     question_text = DEFAULT_IMAGE_TOKEN + sep_token + data['question']
#     # replace_token = DEFAULT_IMAGE_TOKEN
#     # if data_args.mm_use_im_start_end:  # img_st and img_ed
#     #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#     # question_text = question_text.replace(DEFAULT_IMAGE_TOKEN, replace_token)
#     sample_text = question_text + sep_token
#     if is_training:
#         sample_text += data['entity_text'] + eos_token
#     # done for preprocess_multimodal

#     # conversations is a list of one element
#     # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     input_ids = tokenizer_image_token(sample_text, tokenizer, return_tensors='pt').unsqueeze(0)  # insert IMAGE_TOKEN_INDEX into input_ids
#     targets = input_ids.clone()  # 1, L
#     # mask targets with IGNORE_INDEX
#     src_len = len(tokenizer_image_token(question_text, tokenizer))  # -1? TBD, see unit test results
#     targets[:, :src_len] = IGNORE_INDEX
#     return dict(input_ids=input_ids, labels=targets)


class OvenJsonlDataset(Dataset):
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
                no_image=False, # if True, we do not load image; for debug on tructated tokens
                 ) -> None:
        super().__init__()
        self.img_shards_path = img_shards_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.is_training = is_training
        with open(jsonl_path, "r") as f:
            self.data: List[dict] = [json.loads(line) for line in f]
        
        self.is_summary = is_summary
        self.qid_to_summary = None
        if wiki_summary_path is not None:
            with open(wiki_summary_path, "r") as f:
                self.qid_to_summary = json.load(f)  # borrow this to retrieval caching!
        else:  # not providing wiki_summary_path: either do not enable is_summary or in eval
            assert (not self.is_summary) or (not self.is_training), "in training, is_summary is set but no summary is given"

        self.spacy_model = None
        self.max_summary_tokens = max_summary_tokens
        if is_summary or retrieval:
            # handling possible long summary
            self.spacy_model = spacy.load("en_core_web_lg")

        # retrieval related
        self.retrieval = retrieval
        self.entity_image_processor = entity_image_processor
        self.entity_tokenizer = entity_tokenizer
        self.entity_img_path = entity_img_path
        if self.retrieval:
            assert wiki_summary_path is not None, "retrieval requires wiki_summary_path"
            assert DEFAULT_RET_TOKEN in self.tokenizer.get_vocab(), f"retrieval requires {DEFAULT_RET_TOKEN} token in the tokenizer"

        self.qid_to_entity_dict = None
        if qid_to_entity_path is not None:
            with open(qid_to_entity_path, 'r') as f:
                self.qid_to_entity_dict = json.load(f)
        else:
            print("qid_to_entity_path is None, we use entity_text in the dataset; this is not recommended.")

        self.no_image = no_image

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

        if self.is_summary:
            data['entity_summary'] = self.qid_to_summary[entity_id]
        
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
        data_dict = preprocess_oven_with_summary(data, self.tokenizer, self.data_args, 
                                                 is_training=self.is_training, 
                                                 is_retrieval=self.retrieval,  # TODO: add ret in the preprocess; run unit test first.
                                                 max_summary_tokens=self.max_summary_tokens,
                                                 spacy_model=self.spacy_model,
                                                 is_summary=self.is_summary, )
        # remove the redundent batch dim
        data_dict = {k: v.squeeze(0) for k, v in data_dict.items()}
        data_dict['image'] = image
        # not in is_training will drop entity_text
        # you should prepare external labels for eval 
        # (entity / query set, seen / unseen etc.)
        if self.retrieval:
            # collect wiki image
            entity_id = data['entity_id']
            entity_label = torch.LongTensor([int(entity_id[1:])])  # Q123 -> 123
            entity_text = data['entity_text']
            entity_summary = self.qid_to_summary[entity_id]
            entity_summary = get_max_sentences(self.spacy_model, entity_summary, 
                            tokenizer_len_func=lambda x: len(self.entity_tokenizer.encode(x, add_special_tokens=False)), 
                            max_tokens=64,  # CLIP text encoder has 77 context window
                            )
            # concat entity_text and entity_summary
            entity_text_summary = entity_text.strip() + ": " + entity_summary.strip()
            # entity_tokenized = self.entity_tokenizer(entity_text_summary, return_tensors='pt')
            # entity_token_ids = entity_tokenized['input_ids'].squeeze(0)
            # entity_attention_mask = entity_tokenized['attention_mask'].squeeze(0)

            entity_img_path = self.find_entity_img_path(entity_id)
            if entity_img_path is None:
                entity_img = Image.new('RGB', (336, 336), (0, 0, 0))
            else:
                entity_img = Image.open(entity_img_path).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':  # controlled by llava image options
                entity_img = expand2square(entity_img, tuple(int(x*255) for x in self.entity_image_processor.image_mean))
                entity_img = self.entity_image_processor.preprocess(entity_img, return_tensors='pt')['pixel_values'][0]
            else:
                entity_img = self.entity_image_processor.preprocess(entity_img, return_tensors='pt')['pixel_values'][0]
            # update data_dict: remember to manually pop entity_labels in the eval loop
            data_dict.update({
                'entity_image': entity_img,
                'entity_text_summary': entity_text_summary,  # left for collator
                'entity_labels': entity_label, 
            })

        if not self.is_training:
            for k in self.EVAL_NEEDED_KEYS:
                try:
                    data_dict[k] = data[k]
                except KeyError:
                    pass   # missing due to test split
        return data_dict  # wait for data_colltor
    

class RatioSampledDataset(OvenJsonlDataset):
    def __init__(self, sample_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_ratio = sample_ratio
        # build a mixup index list for sampling. we do not introduce randomness here
        # also to avoid bias (and easy impl), we only support INT sampling
        self.mixup_index = list(range(len(self.data))) * self.sample_ratio

    def __len__(self):
        return len(self.mixup_index)

    @property
    def entity_labels(self):  # get all entity labels in entity QID, so that we can build a balanced sampler
        return super().entity_labels * self.sample_ratio
    
    def __getitem__(self, index):
        return super().__getitem__(self.mixup_index[index])
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    entity_tokenizer: transformers.PreTrainedTokenizer = None

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
        if any(k in instances[0] for k in OvenJsonlDataset.EVAL_NEEDED_KEYS):
            for k in OvenJsonlDataset.EVAL_NEEDED_KEYS:
                if k in instances[0]:
                    # grouped by batch for eval
                    batch[k] = [instance[k] for instance in instances]

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)  # ensured by preprocessor
            else:
                batch['images'] = images
        
        # handle possible retrieval batching:
        if self.entity_tokenizer is not None:
            if 'entity_image' in instances[0]:  # only add plural in `image` or `entity_image` keys
                entity_images = [instance['entity_image'] for instance in instances]
                if all(x is not None and x.shape == entity_images[0].shape for x in entity_images):
                    batch['entity_images'] = torch.stack(entity_images)
                else:
                    batch['entity_images'] = entity_images
            entity_labels = [instance["entity_labels"] for instance in instances]
            entity_labels = torch.cat(entity_labels, dim=0)
            # entity_labels do not need truncation;
            # update the batch
            entity_text_inputs = self.entity_tokenizer(
                [instance["entity_text_summary"] for instance in instances], 
                return_tensors='pt', padding=True, truncation=True  # auto padding and truncation to max_length
            )
            batch.update({
                'entity_labels': entity_labels,
                # wait, clip encoder does not have seperate pad_token_id
                # 'entity_attention_mask': entity_token_ids.ne(self.entity_tokenizer.pad_token_id),
                'entity_token_ids': entity_text_inputs['input_ids'],
                'entity_attention_mask': entity_text_inputs['attention_mask'],
            })

        return batch
    

def build_oven_data_modules(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args,
    is_training = True, 
):
    if data_args.multi_val_jsonl_path is not None:
        train_dataset = ConcatDataset([
            OvenJsonlDataset(jsonl_path, tokenizer, data_args, 
            data_args.train_img_shards_path, is_training=is_training, 
            wiki_summary_path=data_args.qid_to_summary_path,
            is_summary=data_args.use_summary,
            max_summary_tokens=data_args.max_summary_tokens,
            retrieval=data_args.retrieval,
            entity_image_processor=data_args.entity_image_processor,
            entity_tokenizer=data_args.entity_tokenizer, 
            entity_img_path=data_args.entity_img_path,
            qid_to_entity_path=data_args.train_qid_to_entity_path,)
        for jsonl_path in data_args.multi_val_jsonl_path]
        )
    # handle multi_val_jsonl_path; with this arg, we are eval on mixture of eval datasets
    else:
        train_dataset = OvenJsonlDataset(data_args.train_jsonl_path, tokenizer, data_args, 
                                     data_args.train_img_shards_path, is_training=is_training, 
                                     wiki_summary_path=data_args.qid_to_summary_path,
                                     is_summary=data_args.use_summary,
                                     max_summary_tokens=data_args.max_summary_tokens,
                                     retrieval=data_args.retrieval,
                                     entity_image_processor=data_args.entity_image_processor,
                                     entity_tokenizer=data_args.entity_tokenizer, 
                                     entity_img_path=data_args.entity_img_path,
                                     qid_to_entity_path=data_args.train_qid_to_entity_path,)
    data_collator = DataCollatorForSupervisedDataset(tokenizer, data_args.entity_tokenizer)  
    # enable entity collate when entity_tokenizer is not None

    if data_args.query_jsonl_path is not None:
        if not isinstance(data_args.query_jsonl_path, list):
            data_args.query_jsonl_path = [data_args.query_jsonl_path]
        if data_args.query_sample_ratio is None:
            data_args.query_sample_ratio = [1] * len(data_args.query_jsonl_path)
        assert len(data_args.query_jsonl_path) == len(data_args.query_sample_ratio)
        print("Loading query dataset from {} with sample_ratio {}...".format(data_args.query_jsonl_path, 
                                                                             data_args.query_sample_ratio))
        train_dataset_list = [train_dataset]

        for i, jsonl_path in enumerate(data_args.query_jsonl_path):
            query_dataset = RatioSampledDataset(sample_ratio=data_args.query_sample_ratio[i],
                jsonl_path=jsonl_path, tokenizer=tokenizer, data_args=data_args,  
                     img_shards_path=data_args.train_img_shards_path, 
                    is_training=is_training, 
                    wiki_summary_path=data_args.qid_to_summary_path,
                is_summary=data_args.use_summary,
                max_summary_tokens=data_args.max_summary_tokens,
                retrieval=data_args.retrieval,
                entity_image_processor=data_args.entity_image_processor,
                entity_tokenizer=data_args.entity_tokenizer, 
                entity_img_path=data_args.entity_img_path,
                qid_to_entity_path=data_args.train_qid_to_entity_path,)
            train_dataset_list.append(query_dataset)
        # query_dataset = RatioSampledDataset(sample_ratio=data_args.query_sample_ratio,
        #     jsonl_path=data_args.query_jsonl_path, tokenizer=tokenizer, data_args=data_args,  
        #          img_shards_path=data_args.train_img_shards_path, 
        #         is_training=is_training, 
        #         wiki_summary_path=data_args.qid_to_summary_path,
        #     is_summary=data_args.use_summary,
        #     max_summary_tokens=data_args.max_summary_tokens,
        #     retrieval=data_args.retrieval,
        #     entity_image_processor=data_args.entity_image_processor,
        #     entity_tokenizer=data_args.entity_tokenizer, 
        #     entity_img_path=data_args.entity_img_path,
        #     qid_to_entity_path=data_args.train_qid_to_entity_path,)
        train_dataset = ConcatDataset(train_dataset_list)

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


if __name__ == '__main__':
    pass