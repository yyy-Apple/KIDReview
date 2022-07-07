import jsonlines
from utils.data_utils import Datum
from typing import List
import random
import numpy as np
import nltk


# common helper functions
def load_data(split):
    data = []
    with jsonlines.open(f'data/{split}.jsonl') as reader:
        for obj in reader:
            datum = Datum.from_json(obj)
            data.append(datum)
    return data


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def read_file_to_list(file_name):
    lines = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f.readlines():
            lines.append(line.strip())
    return lines


def write_list_to_file(list_to_write, filename):
    out_file = open(filename, 'w')
    for line in list_to_write:
        print(line, file=out_file)
    out_file.flush()
    out_file.close()


def read_jsonlines_to_list(file_name):
    lines = []
    with jsonlines.open(file_name, 'r') as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def write_list_to_jsonline(list_to_write, filename):
    with jsonlines.open(filename, 'w') as writer:
        writer.write_all(list_to_write)


def get_sents(text: str) -> (List, List):
    """ give a text string, return the sentence list """
    # Here are some heuristics that we use to get appropriate sentence splitter.
    # 1. Delete sentences that are fewer than 25 characters.
    # 2. If a sentence ends in et al. Then concate with the sentence behind it.
    sent_list: List[str] = nltk.tokenize.sent_tokenize(text)
    new_sent_list = [sent.replace("\n", "") for sent in sent_list]
    postprocessed = []
    buff = ""
    for sent in new_sent_list:
        if sent.endswith('et al.') or sent.endswith('Eq.') \
                or sent.endswith('i.e.') or sent.endswith('e.g.'):
            buff += sent
        else:
            if len(buff + sent) > 25 and \
                    not (buff + sent).__contains__('arxiv') and \
                    not (buff + sent).__contains__('http'):
                postprocessed.append(buff + sent)
            buff = ""
    if len(buff) > 0:
        postprocessed.append(buff)
    return postprocessed[:250]


def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


def sig_test(l1: List, l2: List, perc=0.8, N=10000):
    assert len(l1) == len(l2)
    l1_avg = sum(l1) / len(l1)
    l2_avg = sum(l2) / len(l2)
    fail_num = 0

    indices = np.arange(len(l1))
    for i in range(N):
        samples = random.choices(indices, k=int(len(l1) * perc))

        sub_l1 = [l1[sample] for sample in samples]
        sub_l2 = [l2[sample] for sample in samples]
        sub_l1_avg = sum(sub_l1) / len(sub_l1)
        sub_l2_avg = sum(sub_l2) / len(sub_l2)

        if (l1_avg > l2_avg and sub_l1_avg > sub_l2_avg) or (l1_avg < l2_avg and sub_l1_avg < sub_l2_avg):
            pass
        else:
            fail_num += 1
    print(f'l1_avg: {l1_avg}')
    print(f'l2_avg: {l2_avg}')
    print(f'Significant test, p-value={fail_num / N}')
