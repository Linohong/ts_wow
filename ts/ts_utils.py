'''
    Code by gaulinmp
    https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4

'''
import json
import re
import math
import numpy as np
from itertools import chain
from collections import Counter
import nltk
from nltk.util import ngrams  # This is the ngram magic.
from textblob import TextBlob

NGRAM = 1

re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_alpha = re.compile('[^a-zA-Z]+')
re_stripper_naive = re.compile('[^a-zA-Z\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def get_tuples_nosentences(txt):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    ng = ngrams(re_stripper_alpha.sub(' ', txt).split(), NGRAM)
    return list(ng)


def get_tuples_manual_sentences(txt):
    """Naive get tuples that uses periods or newlines to denote sentences."""
    if not txt: return None
    sentences = (x.split() for x in splitter_naive(txt) if x)
    ng = (ngrams(x, NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))


def get_tuples_nltk_punkt_sentences(txt):
    """Get tuples that doesn't use textblob."""
    if not txt: return None
    sentences = (re_stripper_alpha.split(x) for x in sent_detector.tokenize(txt) if x)
    # Need to filter X because of empty 'words' from punctuation split
    ng = (ngrams(filter(None, x), NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))


def get_tuples_textblob_sentences(txt):
    """New get_tuples that does use textblob."""
    if not txt: return None
    tb = TextBlob(txt)
    ng = (ngrams(x.words, NGRAM) for x in tb.sentences if len(x.words) > NGRAM)
    return [item for sublist in ng for item in sublist]


def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)
    return 1.0 * len(a & b) / len(a | b)


def cosine_similarity_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

def possible_topics(answer_topic, other_topics, threshold=0.5):
    results = []
    a = get_tuples_nosentences(answer_topic.lower().replace('(', '').replace(')', ''))
    for b in other_topics:
        if b == answer_topic:
            continue

        bgtn = get_tuples_nosentences(b.replace('(', '').replace(')', '').lower())
        # print(a, bgtn)
        if cosine_similarity_ngrams(a, bgtn) >= threshold:
            results.append(b)

    return results

def write_ts_result_to_file(filename, pred_record, only_false_positive_cases=False):
    l = len(pred_record)
    print_result = []
    with open(filename, 'w') as f:
        for original_id, sorted_pred_record in pred_record.items():

            cur_result = {'original_id': original_id, 'dialog_history':'', 'cand':[], }
            for i, info_dict in enumerate(sorted_pred_record):
                cur_result['dialog_history'] = info_dict['dialog_history']
                cur_result['cand'].append({
                    'label': info_dict['label'],
                    'topic_cand':info_dict['topic_cand'],
                    'topic_sents':info_dict['topic_sent'],
                    'score': info_dict['score']
                })
            print_result.append(cur_result)

        json.dump(print_result, f, indent=4, sort_keys=False)

    return
# # a = get_tuples_nosentences("History of good science fiction".lower())
# # print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a,b), cosine_similarity_ngrams(a,b)))


