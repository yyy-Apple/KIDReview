# %%
import logging
import os

from args import get_args
from graphbart import GraphBART
from utils.model_utils import *
from utils.utils import *

logging.disable(logging.WARNING)


def set_seed_everywhere(seed, cuda):
    """ Set seed for reproduce """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


set_seed_everywhere(666, True)


def main(args):
    print(args)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    assert os.path.exists(save_dir) and os.path.isdir(save_dir)

    save_model_name = f'bart'
    if args.ref_graph:
        save_model_name += '_ref'
    if args.citation_graph:
        save_model_name += '_cite'
    if args.concept_graph:
        save_model_name += '_concept'
    save_model_name += '.pth'
    save_path = os.path.join(save_dir, save_model_name)

    graphbart = GraphBART(args)

    if args.reload_from_saved is not None:
        graphbart.load_model(args.reload_from_saved)

    for split in ['train', 'val']:
        data = load_data(split)
        graphbart.load_data(set_type=split, data=data)

    train_steps = args.n_epochs * (len(graphbart.train_dataset) // args.batch_size + 1)
    warmup_steps = int(train_steps * args.warmup_proportion)
    graphbart.get_optimizer(
        lr1=args.lr1,
        lr2=args.lr2,
        lr3=args.lr3,
        lr4=args.lr4,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon)

    print(f'Optimizers set.')

    best_loss = 1e8
    for epoch in range(args.n_epochs):
        print(f"On epoch {epoch}")
        graphbart.train_epoch(batch_size=args.batch_size)
        current_loss = graphbart.evaluate()
        print(f'Current loss on dev set is {current_loss}')
        if current_loss < best_loss:
            print(f'Saving best model...')
            graphbart.save_model(f'{save_path}best')
            best_loss = current_loss


if __name__ == '__main__':
    args = get_args()
    main(args)
