from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .transformer import *
from .build import *


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', './clip-tokenizer'
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', './clip-tokenizer'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'])

    return tokenizer