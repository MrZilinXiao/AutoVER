# please run all wiki summary ahead of time, 
# to avoid training interruption
import re


def split_by_punc(text):
    pattern = re.compile(r'(?<=[^\s\w])\s+')
    return pattern.split(text)

def split_sentence(spacy_model, text):
    # split text into sentences
    doc = spacy_model(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_max_sentences(spacy_model, text, tokenizer_len_func, max_tokens=128, 
                      split_type='sent', debug=False):
    # return a prefix string that suits the max_tokens
    if max_tokens is None or spacy_model is None:
        return text

    if text is None:
        return None
    
    if split_type == 'sent':
        sents = split_sentence(spacy_model, text)
    elif split_type == 'punc':
        sents = split_by_punc(text)
    else:   # the most extre case is to split by space
        print("Warning: you are splitting by space, which is unusual")
        if debug:
            print(f"problem text: `{text}`")
        sents = text.split()

    if len(sents) == 0:
        return ""
    
    # ensure we have a trivial solution that the first part always fits
    if tokenizer_len_func(sents[0]) > max_tokens:
        if split_type == 'sent':
            return get_max_sentences(spacy_model, text, 
                                    tokenizer_len_func, max_tokens, 'punc', debug)
        elif split_type == 'punc':
            return get_max_sentences(spacy_model, text, 
                                    tokenizer_len_func, max_tokens, 'space', debug)
        else:  # even split by space does not work; I would say just return and let tokenizer's truc deal with it
            # raise RuntimeError("Typically impossible to get here")
            print(f"The first part of text is too long: `{sents[0]}`")
            return sents[0]
        
    # now we have a trivial solution, 
    # gradually extend to more parts until reaching the max_tokens
    return_str = sents[0]
    for sent_i in range(1, len(sents)):
        if tokenizer_len_func(return_str + sents[sent_i]) > max_tokens:
            break
        else:
            return_str += " " + sents[sent_i]
    return return_str

if __name__ == '__main__':
    import spacy
    from transformers import AutoTokenizer
    spacy_model = spacy.load("en_core_web_lg")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = "This is a test sentence.       This is another test sentence. This is the third test sentence."
    print(
        get_max_sentences(spacy_model, text, 
                          tokenizer_len_func=lambda x: len(tokenizer.encode(x, add_special_tokens=False)), 
                          max_tokens=20)
    )