# custom hard negative mining sampler & funcs moved to here
import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
import itertools
import random

def torch_shuffle(tensors: torch.LongTensor, generator=None):
    """
    Shuffle a tensor along its first dimension.
    """
    if tensors.numel() == 0:
        return tensors
    return tensors[torch.randperm(tensors.size(0), generator=generator)]

def norm_counts_sum_to_one(counts, alpha=0.):
    assert 0 <= alpha <= 1, f"alpha should be in [0, 1], got {alpha}"

    s = sum(counts)
    ori_weights = np.asarray([c / s for c in counts])
    uniform_weights = np.asarray([1 / len(counts) for _ in counts])
    weights = (1 - alpha) * ori_weights + alpha * uniform_weights

    # make sure summing up to 1
    weights = weights / weights.sum()

    return weights


# I know this is ugly, consider more elegant rejection sampling in code release
def get_hard_negative_per_batch_indices(labels: List[int], batch_size, world_size, generator=None, label_to_count=None, total_len=None, 
                                        retry_times=10, hard_negative_ratio=0.5, hard_negative_groups: Dict[int, List[int]] = None,):
    """
    receive labels: provide mapping to get a labels_to_indices dict
    receive hard_negative_groups: a mapping for each entity to possible hard negative entities
    receive hard_negative_ratio: the ratio of hard negative samples in a batch; the rest are from one-class-per-batch
    """
    avail_entities = set(labels)   # Q-prefix are stripped
    avail_hard_entities = set(hard_negative_groups.keys()) & avail_entities
    assert len(avail_hard_entities) > 0, "No hard negative entities available for sampling"

    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)

    for k, v in labels_to_indices.items():
        np.random.shuffle(v)
        labels_to_indices[k] = itertools.cycle(v)

    megabatch_size = world_size * batch_size
    normal_mega_size = int(megabatch_size * (1 - hard_negative_ratio))
    hard_negative_mega_size = megabatch_size - normal_mega_size
    # make hard_negative_mega_size even
    assert hard_negative_mega_size % 2 == 0, "hard_negative_mega_size should be even, adjust your hard_negative_ratio"
    # hard_negative_mega_size not strictly followed, 
    # as hard_negative_groups can be too sparse to provide enough hard negative samples
    indices = []
    used_indices = set()
    padding_count_list = []
    while len(indices) < total_len:
        normal_chosen_entities = np.random.choice(
            list(label_to_count.keys()),
            size=normal_mega_size,
            replace=False,
            p=norm_counts_sum_to_one(list(label_to_count.values()))
        ).tolist()
        hard_chosen_candidates = list(avail_hard_entities - set(normal_chosen_entities))
        hard_chosen_entities = np.random.choice(
            hard_chosen_candidates,
            size=hard_negative_mega_size // 2,
            replace=False,
            p=norm_counts_sum_to_one([label_to_count[ent] for ent in hard_chosen_candidates])
        ).tolist()
        # adding normal samples + hard_chosen_entities first; 
        for i, entity in enumerate(normal_chosen_entities + hard_chosen_entities):
            entity_iterator = labels_to_indices[entity]
            index = next(entity_iterator)
            retry_this_entity = retry_times
            while index in used_indices:
                index = next(entity_iterator)
                retry_this_entity -= 1
                if retry_this_entity == 0:
                    # print(f"Retry times exceeded for entity {entity}; Using SEEN sample again.")
                    break
            # if still in, why not randonly draw another entity instead!
            if index in used_indices:
                if i < len(normal_chosen_entities):  # drawing from normal entities
                    redrawing_entities = np.random.choice(  # considered as second-time rej sampling
                        list(label_to_count.keys()),
                        size=normal_mega_size,
                        replace=False,
                        p=norm_counts_sum_to_one(list(label_to_count.values()))
                    ).tolist()
                else:
                    redrawing_entities = np.random.choice(  # considered as second-time rej sampling
                        hard_chosen_candidates,
                        size=hard_negative_mega_size // 2,
                        replace=False,
                        p=norm_counts_sum_to_one([label_to_count[ent] for ent in hard_chosen_candidates])
                    ).tolist()

                for ent in redrawing_entities:
                    if ent in normal_chosen_entities + hard_chosen_entities:
                        continue
                    ent_iterator = labels_to_indices[ent]
                    index = next(ent_iterator)
                    retry_this_entity = retry_times
                    while index in used_indices:
                        index = next(ent_iterator)
                        retry_this_entity -= 1
                        if retry_this_entity == 0:
                            # print(f"Retry times exceeded for entity {entity}; Using SEEN sample again.")
                            break

                    if index not in used_indices:
                        # in-place update when finding a new sample
                        if i < len(normal_chosen_entities):
                            normal_chosen_entities[i] = ent
                        else:
                            hard_chosen_entities[i - len(normal_chosen_entities)] = ent
                        break

            indices.append(index)
            used_indices.add(index)
        # adding hard negative samples; pad with normal samples
        padding_count = 0
        for hard_ent in hard_chosen_entities:
            hard_ent_candidates = set(hard_negative_groups[hard_ent]) & avail_entities
            # awkward if no or few hard negative candidates available
            if len(hard_ent_candidates) == 0:
                padding_count += 1
                continue
            select_neg_ent = random.choice(list(hard_ent_candidates))

            entity_iterator = labels_to_indices[select_neg_ent]
            index = next(entity_iterator)
            retry_this_entity = retry_times
            while index in used_indices:
                index = next(entity_iterator)
                retry_this_entity -= 1
                if retry_this_entity == 0:
                    # print(f"Retry times exceeded for entity {entity}; Using SEEN sample again.")
                    break
            # consider sparse negative candidates, we allow repeated samples in hard negative part
            indices.append(index)
            used_indices.add(index)
        # the rest are padding; we expect padding entities to be rare, so no rej-sampling
        padding_count_list.append(padding_count)
        if padding_count > 0:
            padding_entities = np.random.choice(
                list(label_to_count.keys()),
                size=padding_count,
                replace=False,
                p=norm_counts_sum_to_one(list(label_to_count.values()))
            ).tolist()
            indices.extend([next(labels_to_indices[entity]) for entity in padding_entities])

        if len(indices) % 10000 == 0:  # 50w for 0.1
            print(f"Generated {len(indices)} indices...")

    # all done, should print some stats about: 1) how many batches has conflicting entities; 2) number of missing samples
    num_mega_batches = total_len // megabatch_size
    conflicting_batches = 0
    for j in range(0, total_len, megabatch_size):
        this_megabatch_indices = indices[j : j + megabatch_size]
        this_megabatch_classes = [labels[i] for i in this_megabatch_indices]
        # assert len(set(this_megabatch_classes)) == megabatch_size, f"Should have only one class per batch. Got {this_megabatch_classes}"
        if len(set(this_megabatch_classes)) != megabatch_size:
            conflicting_batches += 1

    print(f"Number of conflicting batches: {conflicting_batches} out of {num_mega_batches}")
    missing_samples = set(range(total_len)) - set(indices)  # total_len is the size of train_set; indices is the sampled indices
    print(f"Number of training samples: {total_len} or {len(indices)}")
    print(f"Number of missing samples: {len(missing_samples)}")
    print(f"Number of average padding samples: {sum(padding_count_list) / len(padding_count_list)}")

    return indices
        

# abandoned, as this does not cover all sampler as many as possible
def get_one_class_per_batch_indices(labels, batch_size, world_size, generator=None, 
                                    label_to_count=None, total_len=None):
    # WARNING: this will definitely increase underrepresentative samples occurance in the training set
    # as they will be sampled more frequently; not sure if this is a good thing
    megabatch_size = world_size * batch_size

    labels_to_indices = defaultdict(list)  # entity_qid -> query_indices in trainset
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = torch_shuffle(torch.tensor(v, dtype=torch.long))

    label_iterators = {k: itertools.cycle(v.tolist()) for k, v in labels_to_indices.items()}
    # entity_qid -> unlimited query_indices

    # TODO: does this cycle bring repeated samples? yes but inevitable; use rejection sampling after. 
    indices = []

    while len(indices) < total_len:
        chosen_entities = np.random.choice(  # ensure one class per batch by selecting entities
            list(label_to_count.keys()), 
            size=megabatch_size, 
            replace=False, 
            p=norm_counts_sum_to_one(list(label_to_count.values()))   # does sampling by count ensuring minimal missing samples?
        )
        # assert len(set(chosen_entities)) == len(chosen_entities), "Do not allow repeated entities in a batch."
        indices.extend([next(label_iterators[entity]) for entity in chosen_entities])
    # run unit test on indices, see how many samples are missing
    for j in range(0, total_len, megabatch_size):
        this_megabatch_indices = indices[j : j + megabatch_size]
        this_megabatch_classes = [labels[i] for i in this_megabatch_indices]
        assert len(set(this_megabatch_classes)) == megabatch_size, f"Should have only one class per batch. Got {this_megabatch_classes}"

    missing_samples = set(range(total_len)) - set(indices)  # total_len is the size of train_set; indices is the sampled indices
    print(f"Number of training samples: {total_len} or {len(indices)}")
    # Number of training samples: 489037 or 489088
    print(f"Number of missing samples: {len(missing_samples)}")
    # Number of missing samples: 20506
    return indices


def get_one_class_per_batch_indices_v2(labels, batch_size, world_size, generator=None, 
                                    label_to_count=None, total_len=None, retry_times=10):
    """
    v2 function ensures (nearly) each sample goes through the training while keeping one class per batch
    """
    megabatch_size = world_size * batch_size

    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)

    for k, v in labels_to_indices.items():
        np.random.shuffle(v)
        labels_to_indices[k] = itertools.cycle(v)

    indices = []
    used_indices = set()
    while len(indices) < total_len:
        chosen_entities = np.random.choice(
            list(label_to_count.keys()), 
            size=megabatch_size, 
            replace=False, 
            p=norm_counts_sum_to_one(list(label_to_count.values()))
        )

        for entity in chosen_entities:
            entity_iterator = labels_to_indices[entity]
            index = next(entity_iterator)
            retry_this_entity = retry_times
            while index in used_indices:
                index = next(entity_iterator)
                retry_this_entity -= 1
                if retry_this_entity == 0:
                    # print(f"Retry times exceeded for entity {entity}; Using the same sample again.")
                    break
            # if still in, why not randonly draw another entity instead!
            if index in used_indices:
                redrawing_entities = np.random.choice(  # considered as second-time rej sampling
                    list(label_to_count.keys()), 
                    size=megabatch_size, 
                    replace=False, 
                    p=norm_counts_sum_to_one(list(label_to_count.values()))
                )
                for ent in redrawing_entities:
                    if ent in chosen_entities:
                        continue
                    ent_iterator = labels_to_indices[ent]
                    index = next(ent_iterator)
                    retry_this_entity = retry_times
                    while index in used_indices:
                        index = next(ent_iterator)
                        retry_this_entity -= 1
                        if retry_this_entity == 0:
                            # print(f"Retry times exceeded for entity {entity}; Using the same sample again.")
                            break
                    
                    if index not in used_indices:
                        break
                if index in used_indices:
                    print(f"Retry times exceeded for entity {entity} after second-time rej sampling; Using the same sample again.")

            indices.append(index)
            used_indices.add(index)

    num_mega_batches = total_len // megabatch_size
    conflicting_batches = 0
    for j in range(0, total_len, megabatch_size):
        this_megabatch_indices = indices[j : j + megabatch_size]
        this_megabatch_classes = [labels[i] for i in this_megabatch_indices]
        # assert len(set(this_megabatch_classes)) == megabatch_size, f"Should have only one class per batch. Got {this_megabatch_classes}"
        if len(set(this_megabatch_classes)) != megabatch_size:
            conflicting_batches += 1

    print(f"Number of conflicting batches: {conflicting_batches} out of {num_mega_batches}")

    missing_samples = set(range(total_len)) - set(indices)  # total_len is the size of train_set; indices is the sampled indices
    print(f"Number of training samples: {total_len} or {len(indices)}")
    print(f"Number of missing samples: {len(missing_samples)}")
    
    return indices