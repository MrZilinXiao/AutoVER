from typing import Dict, List
from .trie import Trie, DummyTrieEntity, DummyTrieMention
import transformers
import torch
from LLaVA.llava.constants import DEFAULT_RET_TOKEN


STATUS_DEBUG_MESSAGES = {
    'free': 'free-form generation', 
    'entity': 'entity generation',
}

# not all params are useful; for temporary use only
# note that we may have more constrains in future, such as free-form description ahead
# so make it compatible with future changes
def _get_prefix_allowed_tokens_fn_raw(
    encode_fn,
    decode_fn,
    sep_token: str = '\n', 
    sep_token_2: str = None,   # when given, <ret> will serve as the second sep_token
    entity_trie: Trie = None, 
    debug: bool = False,
):
    sep_token_id = [encode_fn(sep_token)[-1]]
    if sep_token_2 is not None:
        sep_token_id.append(encode_fn(sep_token_2)[-1])
    # current sentences would be: 
    # `<image>\n<question>\n`
    # the answer to be expected: `<entity><eos>`; 
    def get_status(sent: List[int]):
        return 'entity'
    
    def get_trie_entity(sent: List[int]):
        # sent is already cropped to the second sep_token_id
        return entity_trie.get(sent)
    
    def find_crop_pos(sent: List[int]):
        # find the second sep_token_id, and assert illegal cases
        indices = [i for i, x in enumerate(sent) if x in sep_token_id]
        assert len(indices) == 2, "num of sep_tokens mismatched, check your constrained_decode_type."
        return indices[1] + 1
    
    def prefix_allowed_tokens_fn(batch_id, sent):
        assert batch_id == 0, f"batch_size = 1 is required in constrained decoding"
        sent = sent.tolist()  # tokenized generated sentence (including inputs)
        # identify the second sep_token pos, and crop from that pos
        sent = sent[find_crop_pos(sent):]
        # get status, currently it's always entity
        status = get_status(sent)
        if debug:
            print(f"status: {STATUS_DEBUG_MESSAGES[status]}")

        if status == 'entity':
            trie_out = get_trie_entity(sent)
        # elif:
        else:
            raise RuntimeError(f"Unknown status: {status}")
        
        if debug:
            print(f"trie_out: {trie_out}, ", end='')
            print(f"which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")
            print('------------------------')

        return trie_out
    return prefix_allowed_tokens_fn


def _get_prefix_allowed_tokens_fn_desc(
    encode_fn,
    decode_fn,
    vocabulary_length,
    bos_token_id,
    pad_token_id, 
    sep_token: str = '\n', 
    entity_trie: Trie = None, 
    debug: bool = False,
):
    sep_token_id = encode_fn(sep_token)[-1]
    dummy_allowed_token_ids = list(
        i for i in range(vocabulary_length) if i not in (bos_token_id, pad_token_id)
    )  # when unconstrained, return this list
    # current sentences would be: 
    # `{image_token}\n{question_intro}{query} {summary_intro}\n`
    # the answer to be expected: `<free-form summary>\n<entity><eos>`; 
    def get_status(sent: List[int]):
        # count how many sep_token_id in sent
        status = sum(x == sep_token_id for x in sent)
        if status <= 1:
            raise RuntimeError(f"impossible num of sep_tokens, check your input format.")
        elif status == 2:  # description part
            return 'free'
        elif status == 3:
            return 'entity'
        else:
            raise RuntimeError(f"too many num of sep_tokens, check your input format.")
    
    def get_trie_entity(sent: List[int]):
        # sent is already cropped to the LAST sep_token_id
        return entity_trie.get(sent)
    
    def find_crop_pos(sent: List[int]):
        # find the LAST sep_token_id, call this func only when status == 'entity'
        indices = [i for i, x in enumerate(sent) if x == sep_token_id]
        assert len(indices) == 3, "num of sep_tokens mismatched, check your constrained_decode_type."
        return indices[-1] + 1
    
    def prefix_allowed_tokens_fn(batch_id, sent):
        assert batch_id == 0, f"batch_size = 1 is required in constrained decoding"
        sent = sent.tolist()  # tokenized generated sentence (including inputs)

        status = get_status(sent)
        if debug:
            print(f"status: {STATUS_DEBUG_MESSAGES[status]}")

        if status == 'free':
            return dummy_allowed_token_ids
        elif status == 'entity':
            # identify the LAST sep_token pos, and crop from that pos
            sent = sent[find_crop_pos(sent):]
            trie_out = get_trie_entity(sent)
        else:
            raise RuntimeError(f"Unknown status: {status}")
        
        if debug:
            print(f"trie_out: {trie_out}, ", end='')
            print(f"which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")
            print('------------------------')

        return trie_out
    
    return prefix_allowed_tokens_fn

# LLaVA adapted simplifed version; does not designed for batch inference.
def get_prefix_allowed_tokens_fn_universal(
    tokenizer: transformers.PreTrainedTokenizer,
    entity_trie: Trie = None,  # can be static from fixed 6M wiki entities
    entity_candidates: List[str] = None,  # or dynamicly retrieved from a knowledge base
    constrained_decode_type: str = 'raw',
    debug: bool = False, 
):
    if entity_trie is None:
        assert entity_candidates is not None, "entity_trie and entity_candidates cannot be both None."
        # TODO: dynamic constructing entity_trie from entity_candidates
        # raise NotImplementedError("Dynamic entity_trie construction is not implemented yet.")
        entity_trie = Trie()
        for entity in entity_candidates:
            entity_trie.add(tokenizer.encode(f"\n{entity}{tokenizer.eos_token}", add_special_tokens=False)[2:])

    if constrained_decode_type.startswith('raw'):   # raw or raw_ret
        return _get_prefix_allowed_tokens_fn_raw(
            lambda x: tokenizer.encode(x, add_special_tokens=False),
            lambda x: tokenizer.decode(torch.tensor(x)),
            sep_token='\n', 
            sep_token_2=DEFAULT_RET_TOKEN if DEFAULT_RET_TOKEN in tokenizer.get_vocab() else None,  
            # when <ret> is in vocab, use it as the second sep_token; this will test ret_version's normal behavior
            entity_trie=entity_trie, 
            debug=debug,
        )
    # desc branch gets abandoned for now
    elif constrained_decode_type == 'desc':
        return _get_prefix_allowed_tokens_fn_desc(
            lambda x: tokenizer.encode(x, add_special_tokens=False),
            lambda x: tokenizer.decode(torch.tensor(x)),
            vocabulary_length=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sep_token='\n', 
            entity_trie=entity_trie, 
            debug=debug,
        )
    else:
        raise NotImplementedError(f"constrained_decode_type {constrained_decode_type} is not implemented yet.")