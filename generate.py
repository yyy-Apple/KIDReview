import logging
import os

from tqdm import trange

from args import get_args
from graphbart import GraphBART
from utils.utils import *

logging.disable(logging.WARNING)


def main(args):
    print(args)
    # load model
    graphbart = GraphBART(args)

    model_path = os.path.join(args.save_dir, args.model_name)

    # TODO: Uncomment this later!!!
    graphbart.load_model(model_path)

    # load data
    data = load_data('unique_test')
    graphbart.load_data(set_type='test', data=data)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    assert os.path.exists(args.output_dir) and os.path.isdir(args.output_dir)
    gold_file_path = os.path.join(args.output_dir, 'gold.txt')
    pred_file_path = os.path.join(args.output_dir, 'pred.txt')
    gold_file = open(gold_file_path, 'w')
    pred_file = open(pred_file_path, 'w')

    test_dataset = graphbart.dataset['test']
    for i in trange(0, len(test_dataset)):
        gold_txt = data[i].raw_review
        pred_txt = graphbart.generate(test_dataset[i])
        print(gold_txt, file=gold_file)
        print(pred_txt[0], file=pred_file)
    gold_file.flush()
    pred_file.flush()
    gold_file.close()
    pred_file.close()


if __name__ == '__main__':
    args = get_args()
    main(args)
