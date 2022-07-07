from argparse import Namespace
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='GraphBART parameters')

    # Model
    parser.add_argument('--attn_head', default=4, type=int, help='Attention head number in GAT')
    parser.add_argument('--graph_hidden_size', default=200, type=int, help='Hidden size in GAT')
    parser.add_argument('--ffn_drop', default=0.1, type=float,
                        help='Drop out probability of feed forward NN in GraphTransformer')
    parser.add_argument('--attn_drop', default=0.1, type=float, help='Attention dropout probability in GAT')
    parser.add_argument('--drop', default=0.1, type=float, help='Regular dropout probability')
    parser.add_argument('--prop', default=2, type=int, help='Number of GAT layers in GraphTransformer')
    parser.add_argument('--checkpoint', default='facebook/bart-large-cnn', type=str,
                        help='The pretrained model to load as a starting point')
    parser.add_argument('--knowledge_first', action='store_true', default=False,
                        help='Whether to first attend to knowledge, then attend to text')
    parser.add_argument('--prepend', action='store_true', default=False,
                        help='Whether to prepend the reference embedding and citation embedding.'
                             'If not set, Concatenate the embeddings with token embeddings, then'
                             'go through a reduction matrix.')
    parser.add_argument('--no_entity_type', action='store_true', default=False,
                        help='Whether to use entity type information.')
    parser.add_argument('--prepend_concept', action='store_true', default=False,
                        help='Whether to prepend entity representations to the encoder output')
    # Training
    parser.add_argument('--source', default=None, required=True, type=str,
                        help='Source of input, could be intro/ext/abs_ext/oracle')
    parser.add_argument('--reload_from_saved', default=None, type=str, help='Reload form saved model')
    parser.add_argument('--gpu', default=2, type=int, help='Number of GPU to use')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr1', default=4e-5, type=float, help='Learning rate for BART')
    parser.add_argument('--lr2', default=0.0001, type=float, help='Learning rate for Graph')
    parser.add_argument('--lr3', default=0.0001, type=float, help='Learning rate for knowledge cross attn')
    parser.add_argument('--lr4', default=4e-5, type=float, help='Learning rate for converting reference'
                                                                'embedding and citation embedding.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--save_dir', default='trained_models', type=str, help='Directory to save trained models')
    parser.add_argument('--ref_graph', action='store_true', default=False, help='Whether to use reference graph')
    parser.add_argument('--citation_graph', action='store_true', default=False, help='Whether to use citation graph')
    parser.add_argument('--concept_graph', action='store_true', default=False, help='Whether to use concept graph')
    parser.add_argument('--concept_graph_global', action='store_true', default=False,
                        help='Whether to use the global node of concept graph')
    parser.add_argument('--src_max_length', default=1024, type=int, help='Input maximum length of BART')
    parser.add_argument('--tgt_max_length', default=1024, type=int, help='Output maximum length of BART')
    parser.add_argument('--cache_dir', default='cache', type=str, help='Directory to save loaded data')
    parser.add_argument('--n_epochs', default=10, type=int, help='Number of training epochs')

    # Generation
    parser.add_argument('--model_name', default='', type=str, help='The saved model name used for generation')
    parser.add_argument('--output_dir', default='output', type=str, help='Directory to save generation results')
    parser.add_argument('--beam', default=4, type=int, help='Beam size for decoding')
    parser.add_argument('--lenpen', default=2.0, type=float, help='Length penalty for decoding')
    parser.add_argument('--max_len', default=1024, type=int, help='Maximum generation length')
    parser.add_argument('--min_len', default=100, type=int, help='Minimum generation length')
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)

    args = parser.parse_args()

    # Constant
    args.emb_dim = 128
    args.hidden_size = 1024
    args.ent_type_num = 6
    args.rel_type_num = 15

    return args


def get_namespace_args():
    args = Namespace(
        emb_dim=128,
        ent_type_num=6,
        rel_type_num=15,
        hidden_size=1024,
        attn_head=8,
        ffn_drop=0.1,
        attn_drop=0.1,
        drop=0.1,
        prop=2,
        checkpoint='sshleifer/distilbart-cnn-6-6',
        knowledge_first=False,
        prepend=True,
        gpu=0,
        batch_size=32,
        lr1=4e-5,
        lr2=0.1,
        lr3=4e-5,
        lr4=0.1,
        adam_epsilon=1e-8,
        weight_decay=0.,
        warmup_proportion=0.1,
        save_dir='trained_models',
        ref_graph=False,
        citation_graph=False,
        concept_graph=True,
        src_max_length=1024,
        tgt_max_length=1024,
        cache_dir='cache',
        n_epochs=20,
        model_name='',
        output_dir='backup/output',
        beam=4,
        lenpen=2.0,
        max_len=1024,
        min_len=100,
        no_repeat_ngram_size=3
    )

    return args
