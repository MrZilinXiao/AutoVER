# see from InsGENEL and llava serving; run unit test in notebook first then copy to here
# this will be a simplied version of InsGENEL
from typing import Optional, List, Mapping, Dict
from .trie import Trie
from .vqa_prefix_utils import get_prefix_allowed_tokens_fn_universal
import torch
import torch.nn.functional as F
import transformers
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_RET_TOKEN, DEFAULT_ENT_CLS_TOKEN, 
    IGNORE_INDEX
)
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token

import requests
from io import BytesIO
from PIL import Image
import re
import warnings

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class FuncWithDebug:
    # this allows `transformers` to print internal activations
    def __init__(self, some_func, debug) -> None:
        self.some_func = some_func
        self.debug = debug
        
    def __call__(self, *args, **kwargs):
        return self.some_func(*args, **kwargs)
    
class UniversalConstrainedDecoder:
    # passing loaded model saves time in jupyter debugging
    # in future support description generation + [RET] token
    # `INPUT_PROMPT` should align well with `common_utils/oven_data.py`
    QUESTION_INTRO = "Question: "
    SUMMARY_INTRO = "Possible Entity Description: "  
    INPUT_PROMPT = {
        "no": "{image_token}\n{query}\n", 
        "raw": "{image_token}\n{query}\n",
        "cls": "{image_token}\n{query}{cls_token}",
        "desc": "{image_token}\n{question_intro}{query} {summary_intro}\n", # expect `{desc}\n{entity_text}{eos}`
        "raw_ret": "{image_token}\n{query}{ret_token}",
        "desc_ret": "{image_token}\n{question_intro}{query} {summary_intro}{ret_token}", # retrieval replaces \n with <ret>
    }
    # CONSTRAINED_DECODE_TYPE = ('no', 'raw', 'cls', 'desc', 'raw_ret', 'desc_ret')
    DEFAULT_RET_TOKEN_ID = 32000
    DEFAULT_CLS_TOKEN_ID = 32001

    def __init__(self, model: transformers.GenerationMixin, tokenizer: transformers.PreTrainedTokenizer,
                 image_processor, 
                 entity_trie: Optional[Trie] = None, 
                 qid_to_entity_dict: Optional[Mapping] = None, 
                 entity_embeddings: Optional[torch.Tensor] = None, 
                 entity_qids: Optional[List[str]] = None, 
                 ) -> None:
        self.CONSTRAINED_DECODE_TYPE = tuple(self.INPUT_PROMPT.keys())
        # do eval here or outside? we may return qid & entity_title at the same_time
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.entity_trie = entity_trie  # if None, falling back to dynamic trie
        self.qid_to_entity_dict = qid_to_entity_dict
        self.entity_to_qid_dict = {v: k for k, v in qid_to_entity_dict.items()}
        if entity_embeddings is not None:
            assert entity_embeddings.shape[0] == len(entity_qids), f"entity_embeddings.shape[0] != len(entity_qids), {entity_embeddings.shape[0]} != {len(entity_qids)}"
            self.entity_embeddings = entity_embeddings  # num_entities, shared_dim; do L2 norm in advance please!
            self.entity_qids = entity_qids

        # get a \n token id
        self.NEWLINE_TOKEN_ID = self.tokenizer.encode('\n', add_special_tokens=False)[-1]

    @torch.inference_mode()
    def entity_retrieval_generate(self, model_args, queries: List[str] = None, input_ids: torch.Tensor = None, 
                                  img_paths: List[str] = None, img_tensors: torch.Tensor = None, 
                                  max_new_tokens=256, num_beams=2, skip_special_tokens=False, debug=False, 
                                  return_raw=False, 
                                  constrained_decode_type='raw_ret',
                                  **model_generation_kwargs,   
                                  # possible entity labels in it; in the order of entity_embeddings
                                  ) -> List[Dict[str, str]]:
        ret = {}
        entity_labels = model_generation_kwargs.pop('entity_labels', None)
        num_entity_candidates = model_generation_kwargs.pop('num_entity_candidates', 100)  # default the top-100 entities
        return_full_scores = model_generation_kwargs.pop('return_full_scores', False)
        insert_correct = model_generation_kwargs.pop('insert_correct', False)  # only for debug

        if entity_labels is None:
            warnings.warn("entity_labels is None, so no top@k will be computed")
        # if provide entity_labels, we return the exact top-k for 
        if queries is not None:
            assert len(queries) == 1, "currently only support batch_size=1 inference"
            format_kwargs = {}
            if constrained_decode_type.startswith('desc'):
                format_kwargs['question_intro'] = self.QUESTION_INTRO
                format_kwargs['summary_intro'] = self.SUMMARY_INTRO
            queries = [
                self.INPUT_PROMPT[constrained_decode_type].format(
                    image_token=DEFAULT_IMAGE_TOKEN, 
                    query=query, 
                    ret_token=DEFAULT_RET_TOKEN, 
                    **format_kwargs
                ) for query in queries
            ]
            input_ids = torch.stack([
                tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                for input_prompt in queries
            ], dim=0)
        else:  # eval_script follows dataset formatting...
            assert input_ids.shape[0] == 1, "currently only support batch_size=1 inference"
            input_ids = input_ids.clone()
        assert self.DEFAULT_RET_TOKEN_ID in input_ids[0], f"input_ids should contain {self.DEFAULT_RET_TOKEN_ID}, got {input_ids[0]}"
        
        if img_tensors is None:  # only for easy debugging in notebook
            imgs = [load_image(img_path) for img_path in img_paths]  
            img_tensors = process_images(imgs, self.image_processor, model_args)  # use `image_aspect_ratio` in model_args
            img_tensors = img_tensors.to(dtype=self.model.dtype)
        else:
            img_tensors = img_tensors.clone()

        assert self.DEFAULT_RET_TOKEN_ID == input_ids[0][-1], f"last token_id should be {self.DEFAULT_RET_TOKEN_ID}, got {input_ids[0][-1]}"
        # begin forward the model; get the hidden state of <ret> (last token)
        first_outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            images=img_tensors.to(self.model.device),
        )
        first_hidden_states = first_outputs.hidden_states   # 1, L, D
        # project to the shared embedding space for retrieval
        ret_token_features = self.model.get_ret_token_projector()(first_hidden_states[0, -1:, :])  # 1, shared_dim
        ret_token_features = F.normalize(ret_token_features, dim=-1)  # 1, shared_dim
        # if debug:
        #     print(f"ret_token_features: {ret_token_features}")
        # use this feature to retrieve entities; in future change to faiss or other efficient retrieval
        ret_scores = ret_token_features @ self.entity_embeddings.T  # 1, num_entities; each row is the matching score for each entity
        # print(ret_scores.shape)
        # get ordered predicted entities
        _, ret_indices = torch.topk(ret_scores, k=ret_scores.shape[-1], dim=-1, sorted=True)  # 1, num_entities
        entity_id_str = None
        if entity_labels is not None:
            entity_id_str = "Q" + str(entity_labels[0].item())
            gt_index = self.entity_qids.index(entity_id_str)
            pred_loc = torch.where(ret_indices[0] == gt_index)[0].item()   # the location of the gt entity in the predicted entities
            # closer to 0 means better; return for out top-k calculation
            ret['pred_loc'] = pred_loc
        if return_full_scores:
            ret['top_scores'] = [(self.qid_to_entity_dict[self.entity_qids[index.item()]], ret_scores[0, index].item()) 
                                  for index in ret_indices[0][:num_entity_candidates]]
            ret['full_scores'] = ret_scores[0, ret_indices[0]]
        
        # gather the top-k entities for constrained decoding
        ret_indices = ret_indices[:, :num_entity_candidates]  # 1, num_entity_candidates
        entity_candidates = [self.entity_qids[ret_index.item()] for ret_index in ret_indices[0]]
        if insert_correct:
            assert entity_id_str is not None, "entity_id_str should not be None when you try to insert correct in candidates"
            entity_candidates.append(entity_id_str)
        
        entity_candidates = [self.qid_to_entity_dict[qid] for qid in entity_candidates]

        if debug and entity_labels is not None:
            print(f"gt_entity: {entity_id_str}, gt_top_pos: {ret['pred_loc']}")
        
        # prepare prefix_allowed_tokens_fn
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn_universal(
            self.tokenizer, 
            None,
            entity_candidates=entity_candidates,
            constrained_decode_type=constrained_decode_type, 
            debug=debug
        )

        prefix_allowed_tokens_fn = FuncWithDebug(prefix_allowed_tokens_fn, debug)
        output_ids = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            images=img_tensors.to(self.model.device),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            **model_generation_kwargs
        )
        if return_raw:
            return output_ids
        elif debug:
            print(f"raw output_ids: {output_ids}")
            # decode output_ids; skip <bos> <image_token>
            print(f"raw outputs: {self.tokenizer.decode(output_ids[0][2:], skip_special_tokens=skip_special_tokens)}")
        
        outputs = [self.tokenizer.decode(output_id[input_ids[i].shape[0]: ], skip_special_tokens=skip_special_tokens) 
                   for i, output_id in enumerate(output_ids)]
        
        if debug:
            print(f"output_ids: {output_ids}")
            print(f"outputs: {outputs}")

        output_entities = [output.split(self.tokenizer.eos_token)[0]
                            for output in outputs]
        try:
            output_qids = [self.entity_to_qid_dict[entity] for entity in output_entities]
        except KeyError:
            output_qids = ['INVALID'] * len(output_entities)
            print(f"BUGS: {output_entities}, as such QIDs have been set to INVALID.")
        
        return [
            {'entity_text': output_entities[0], 'entity_id': output_qids[0], 'entity_desc': 'desc_not_enabled', 
             **ret}
        ]
    
    @torch.inference_mode()
    def entity_classification(self, model_args, queries: List[str] = None, input_ids: torch.Tensor = None, 
                                  img_paths: List[str] = None, img_tensors: torch.Tensor = None, 
                                  max_new_tokens=256, num_beams=2, skip_special_tokens=False, debug=False, 
                                  return_raw=False, 
                                  constrained_decode_type='cls',
                                  **model_generation_kwargs,):
        # pop out the qid_to_cls dict for correct qid return
        cls_to_qid_dict = model_generation_kwargs.pop('cls_to_qid_dict', None)  # reversed dict of `trie_bins/qid_to_cls.json`
        assert cls_to_qid_dict is not None, "cls_to_qid_dict should not be None"

        if queries is not None:
            assert len(queries) == 1, "currently only support batch_size=1 inference"
            queries = [
                self.INPUT_PROMPT[constrained_decode_type].format(
                    image_token=DEFAULT_IMAGE_TOKEN, 
                    query=query, 
                    cls_token=DEFAULT_ENT_CLS_TOKEN, 
                ) for query in queries
            ]
            input_ids = torch.stack([
                tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                for input_prompt in queries
            ], dim=0)
        else:  # eval_script follows dataset formatting...
            assert input_ids.shape[0] == 1, "currently only support batch_size=1 inference"
            input_ids = input_ids.clone()
        # assert self.DEFAULT_RET_TOKEN_ID in input_ids[0], f"input_ids should contain {self.DEFAULT_RET_TOKEN_ID}, got {input_ids[0]}"
        
        if img_tensors is None:  # only for easy debugging in notebook
            imgs = [load_image(img_path) for img_path in img_paths]  
            img_tensors = process_images(imgs, self.image_processor, model_args)  # use `image_aspect_ratio` in model_args
            img_tensors = img_tensors.to(dtype=self.model.dtype)
        else:
            img_tensors = img_tensors.clone()

        assert self.DEFAULT_CLS_TOKEN_ID == input_ids[0][-1], f"last token_id should be {self.DEFAULT_CLS_TOKEN_ID}, got {input_ids[0][-1]}"
        # begin forward the model; get the hidden state of <ent_cls> (last token)
        # construct a label to indicate where the cls_token lies
        labels = IGNORE_INDEX * torch.ones_like(input_ids)
        labels[:, -1] = 0   # give a fixed label to avoid errors

        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            images=img_tensors.to(self.model.device),
            labels=labels.to(self.model.device),
        )
        pred_qid = cls_to_qid_dict[outputs.pred[0].item()]
        cls_logits = outputs.logits[0]
        # compute confidence on cls_logits using softmax
        cls_probs = F.softmax(cls_logits, dim=-1)
        cls_conf = cls_probs[outputs.pred[0].item()].item()

        print(f"pred_qid: {pred_qid}; pred_ent: {self.qid_to_entity_dict[pred_qid]}; conf: {cls_conf}")
        # organize to return format
        return [{'entity_text': self.qid_to_entity_dict[pred_qid], 'entity_id': pred_qid, 'entity_desc': "desc_not_enabled"}]

    @torch.inference_mode()
    def entity_generate(self, model_args, queries: List[str] = None, input_ids: torch.Tensor = None, 
                        img_paths: List[str] = None, img_tensors: torch.Tensor = None, 
                        max_new_tokens=256, num_beams=2, skip_special_tokens=False, debug=False, 
                        return_raw=False, 
                        constrained_decode_type='raw',
                        **model_generation_kwargs, 
                        ) -> List[Dict[str, str]]:
        # currently support batch_size=1 inference
        assert queries is not None or input_ids is not None, "either queries or input_ids should be provided"
        assert img_paths is not None or img_tensors is not None, "either img_paths or img_tensors should be provided"
        assert constrained_decode_type in self.CONSTRAINED_DECODE_TYPE, f"constrained_decode_type should be one of {self.CONSTRAINED_DECODE_TYPE}, got {constrained_decode_type}"
        
        # forward to retrieval_generate if constrained_decode_type is ret
        if constrained_decode_type.endswith('ret'):
            return self.entity_retrieval_generate(model_args, queries, input_ids, img_paths, img_tensors,
                                                    max_new_tokens, num_beams, skip_special_tokens, debug,
                                                    return_raw, constrained_decode_type, **model_generation_kwargs)
        elif constrained_decode_type == 'cls':   # classifier-based VER, need a cls_to_qid dict
            return self.entity_classification(model_args, queries, input_ids, img_paths, img_tensors,
                                                    max_new_tokens, num_beams, skip_special_tokens, debug,
                                                    return_raw, constrained_decode_type, **model_generation_kwargs)

        if queries is not None:
            assert len(queries) == 1, "currently only support batch_size=1 inference"
            format_kwargs = {}
            if constrained_decode_type.startswith('desc'):
                format_kwargs['question_intro'] = self.QUESTION_INTRO
                format_kwargs['summary_intro'] = self.SUMMARY_INTRO
            queries = [
                self.INPUT_PROMPT[constrained_decode_type].format(  # no, raw or desc
                    image_token=DEFAULT_IMAGE_TOKEN, 
                    query=query, 
                    **format_kwargs
                ) for query in queries
            ]

            input_ids = torch.stack([
                tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                for input_prompt in queries
            ], dim=0)
        else:  # eval_script follows dataset formatting...
            assert input_ids.shape[0] == 1, "currently only support batch_size=1 inference"
            input_ids = input_ids.clone()

        # sometimes you test RET-model using raw setting, so if we find <ret> in the tokenizer, we update the input_prompt
        if DEFAULT_RET_TOKEN in self.tokenizer.get_vocab() and input_ids[0][-1] != self.DEFAULT_RET_TOKEN_ID:
            assert input_ids[0][-1] == self.NEWLINE_TOKEN_ID, f"last token_id should be {self.NEWLINE_TOKEN_ID}, got {input_ids[0][-1]}"
            input_ids[0][-1] = self.DEFAULT_RET_TOKEN_ID

        if img_tensors is None:  # only for easy debugging in notebook
            imgs = [load_image(img_path) for img_path in img_paths]  
            img_tensors = process_images(imgs, self.image_processor, model_args)  # use `image_aspect_ratio` in model_args
            img_tensors = img_tensors.to(dtype=self.model.dtype)
        else:
            img_tensors = img_tensors.clone()
        # prepare prefix_allowed_tokens_fn
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn_universal(
            self.tokenizer, 
            self.entity_trie,
            constrained_decode_type=constrained_decode_type, 
            debug=debug
        )
        prefix_allowed_tokens_fn = FuncWithDebug(prefix_allowed_tokens_fn, debug)
        # don't need KeywordStop; during training we use tokenizer.eos_token
        # now the model always greedily decode, 
        # but we expect some randomness in description generation
        output_ids = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            images=img_tensors.to(self.model.device),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if constrained_decode_type != 'no' else None,
            **model_generation_kwargs
        )
        if return_raw:
            return output_ids
        elif debug:
            print(f"raw output_ids: {output_ids}")
            # decode output_ids; skip <bos> <image_token>
            print(f"raw outputs: {self.tokenizer.decode(output_ids[0][2:], skip_special_tokens=skip_special_tokens)}")
        
        outputs = [self.tokenizer.decode(output_id[input_ids[i].shape[0]: ], skip_special_tokens=skip_special_tokens) 
                   for i, output_id in enumerate(output_ids)]
        if debug:
            print(f"output_ids: {output_ids}")
            print(f"outputs: {outputs}")
        # output: `<optional description><sep><possible_ret><some entity><sep>` 
        # like `Gorkhi-Terelj National Park</s>`
        # this may aligned more well with llava! 
        # parse every output and map to QID, though we return them all
        # if constrained_decode_type in ('no', 'raw'):
        output_descs = ["desc_not_enabled"] * len(outputs)
        if constrained_decode_type in ('no', 'raw', 'raw_ret'):
            output_entities = [output.split(self.tokenizer.eos_token)[0]
                               for output in outputs]
        elif constrained_decode_type in ('desc', 'desc_ret'):
            output_entities = [output.split('\n')[-1].replace(self.tokenizer.eos_token, '')
                               for output in outputs]
            try:
                output_descs = [output.split('\n')[-2] for output in outputs]
            except IndexError:  # sometimes in desc_eval, this happens along with INVALID entity extraction
                # we don't touch it for now
                output_descs = ["desc extraction failed"] * len(outputs)

        # qid matching could fail due to bugs; print them when happened
        try:
            output_qids = [self.entity_to_qid_dict[entity] for entity in output_entities]
        except KeyError:
            output_qids = ['INVALID'] * len(output_entities)
            print(f"BUGS: {output_entities}, as such QIDs have been set to INVALID.")
        
        return [
            {'entity_text': entity_text, 'entity_id': qid, 'entity_desc': entity_desc}
            for entity_text, qid, entity_desc in zip(output_entities, output_qids, output_descs)
        ]