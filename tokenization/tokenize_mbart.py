"""Collects the text corpus from given feature files.
"""
import gzip
import os
import pickle
from argparse import ArgumentParser

from transformers import MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')


def main(args):
    for dataset in ['dev', 'test', 'train']:
        print(dataset)
        feature_file = os.path.join(args.data_root, f'{args.feature_name}.{dataset}')
        with gzip.open(feature_file, 'rb') as ff:
            samples = pickle.load(ff)
            new_feature_file = os.path.join(args.data_root, f'{args.new_feature_name}.{dataset}')
            with gzip.open(new_feature_file, 'wb') as of:
                with tokenizer.as_target_tokenizer():
                    for i, sample in enumerate(samples):
                        text = sample['text']
                        token_ids = tokenizer(text)['input_ids']
                        tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
                        # tokens[0]: de_DE
                        # tokens[-1]: </s>
                        # these need to be removed
                        assert tokens[0] == 'de_DE'
                        assert tokens[-1] == '</s>'
                        tokens = tokens[1:-1]
                        token_string = ' '.join(tokens)
                        sample['text'] = token_string
                    pickle.dump(samples, of)


if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument('--data_root', required=True)
    argparser.add_argument('--feature_name', required=True)
    argparser.add_argument('--new_feature_name', required=True)

    main(argparser.parse_args())
