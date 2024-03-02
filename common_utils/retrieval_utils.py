# most utils are for eval retrieval
import torch
import torch.nn.functional as F
from PIL import Image
from common_utils.oven_data import expand2square
from common_utils.split_utils import get_max_sentences
from torch.utils.data import Dataset, DataLoader, Sampler

from typing import List, Sequence, Dict, Any
import spacy
from tqdm import tqdm
from dataclasses import dataclass
import transformers
import os
import torch.distributed as dist
import math

POSSIBLE_IMG_EXT = ('.jpg', '.JPEG')

def find_entity_img_path(entity_id: str, entity_img_path: str):
    shard_id = entity_id[:4]  # Q1, Q11, Q111 all covers
    for ext in POSSIBLE_IMG_EXT:
        img_path = os.path.join(entity_img_path, f"{shard_id}", f"{entity_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None   # missing img is possible; we should fill a black image


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

class OVENImageEmbeddingDataset(Dataset):

    POSSIBLE_IMG_EXT = ('.jpg', '.JPEG')  # checked
    def __init__(self, oven_id_list, processor, padding=True):
        self.oven_id_list = oven_id_list
        self.processor = processor
        self.padding = padding

    def __len__(self):   # this is okay; ordered sampler will take care of it
        return len(self.oven_id_list)
    
    def find_img_path(self, image_id: str):
        shard_id = image_id.split('_')[1][:2]
        for ext in self.POSSIBLE_IMG_EXT:
            img_path = os.path.join(self.img_shards_path, f"{shard_id}", f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        raise RuntimeError(f"Image {image_id} not found in {img_path}")

    def __getitem__(self, index):
        image_id = self.oven_id_list[index]
        img_path = self.find_img_path(image_id)
        image = Image.open(img_path).convert('RGB')
        processor = self.processor

        if self.padding:
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return dict(
            image=image,
            image_id=image_id,
        )
    

class OVENImageCollator:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        batch = dict()
        batch['image_id'] = [instance['image_id'] for instance in instances]  # as a list of strings

        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

        return batch

class CLIPEmbeddingDataset(Dataset):
    def __init__(self, 
                 qid_to_entity_dict, 
                 data_args, 
                 qid_to_summary
                 ) -> None:
        self.qid_to_entity_dict = qid_to_entity_dict
        self.qids = list(qid_to_entity_dict.keys())
        self.qid_to_summary = qid_to_summary
        self.data_args = data_args
        self.entity_image_processor = data_args.entity_image_processor
        self.entity_tokenizer = data_args.entity_tokenizer
        self.spacy_model = spacy.load("en_core_web_lg")

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, index):
        entity_qid = self.qids[index]
        entity_text = self.qid_to_entity_dict[entity_qid]
        # prepare entity image
        # img_path = self.train_dataset.find_entity_img_path(entity_qid)
        img_path = find_entity_img_path(entity_qid, self.data_args.entity_img_path)
        if img_path is None:
            image = Image.new('RGB', (336, 336), (0, 0, 0))
        else:
            image = Image.open(img_path).convert('RGB')
        # always pad-style
        image = expand2square(image, tuple(int(x*255) for x in self.entity_image_processor.image_mean))
        image = self.entity_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # prepare entity text
        entity_summary = self.qid_to_summary[entity_qid]
        entity_summary = get_max_sentences(self.spacy_model, entity_summary, 
            tokenizer_len_func=lambda x: len(self.entity_tokenizer.encode(x, add_special_tokens=False)), 
            max_tokens=64,  # CLIP text encoder has 77 context window
        )
        entity_text_summary = entity_text.strip() + ": " + entity_summary.strip()
        # collate in data_collator; do not tokenize here
        return dict(
            image=image,
            entity_id=entity_qid,
            entity_text=entity_text,
            entity_text_summary=entity_text_summary,
        )
    
@dataclass
class CLIPEmbeddingCollator:
    entity_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        batch['entity_text'] = [instance['entity_text'] for instance in instances]  # as a list of strings
        batch['entity_id'] = [instance['entity_id'] for instance in instances]  # as a list of strings

        entity_images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == entity_images[0].shape for x in entity_images):
            batch['entity_images'] = torch.stack(entity_images)
        else:
            batch['entity_images'] = entity_images
        
        entity_text_inputs = self.entity_tokenizer(
            [instance["entity_text_summary"] for instance in instances], 
            return_tensors='pt', padding=True, truncation=True  # auto padding and truncation to max_length
        )
        batch.update({
            'entity_token_ids': entity_text_inputs['input_ids'],
            'entity_attention_mask': entity_text_inputs['attention_mask'],
        })

        return batch
    

# @torch.inference_mode()
# def cache_entities_in_batch(model, train_dataset, qid_to_entity_dict, data_args, 
#                    read_from_disk=None, cache_to_disk=None, batch_size=128, num_workers=4) -> torch.Tensor:
#     """
#     Cache the entity embeddings for all qids in qid_to_entity_dict; 
#     Do not iterate train_dataset, it only provides qid_to_summary & helper function
#     return a tensor of shape (num_entities, shared_dim)
#     """
#     if read_from_disk is not None:  # possible directly read from disk
#         ckpt = torch.load(read_from_disk, map_location=model.device)   # serilized list can be on GPUs!
#         return ckpt['entity_tensors'], ckpt['entity_names']
    
#     # prepare cache dataset & collator
#     cache_dataset = CLIPEmbeddingDataset(train_dataset, qid_to_entity_dict, data_args)
#     cache_collator = CLIPEmbeddingCollator(data_args.entity_tokenizer)
#     dataloader = DataLoader(cache_dataset, batch_size=batch_size, 
#                             collate_fn=cache_collator, shuffle=False, num_workers=num_workers)
#     ret_entity_tensors = []
#     ret_entity_names: List[str] = []

#     for raw_batch in tqdm(dataloader, total=len(dataloader)):
#         batch = {k: v.to(model.device) for k, v in raw_batch.items() if isinstance(v, torch.Tensor)}
#         batch['entity_images'] = batch['entity_images'].to(dtype=model.dtype)

#         entity_features = model.get_entity_encoder()(**batch)
#         entity_features = F.normalize(entity_features, dim=-1)

#         ret_entity_tensors.append(entity_features.cpu())
#         ret_entity_names.extend(raw_batch['entity_text'])

#     ret_entity_tensors = torch.cat(ret_entity_tensors, dim=0)

#     if cache_to_disk is not None:
#         torch.save({
#             'entity_tensors': ret_entity_tensors,
#             'entity_names': ret_entity_names,
#         }, cache_to_disk)
#     return ret_entity_tensors, ret_entity_names


if __name__ == '__main__':
    # unit test
    pass