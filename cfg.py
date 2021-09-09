import argparse


def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ds", type=str, default="SOP", choices=["SOP", "CUB"])
    parser.add_argument("--bs", type=int, default=380)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--ep", type=int, default=5)
    parser.add_argument("--qlen", type=int, default=32)
    parser.add_argument("--hyp_c", type=float, default=0.1)
    parser.add_argument("--no_hyp", dest="hyp", action="store_false")
    parser.add_argument("--eval_ep", type=int, default=4)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    return parser.parse_args()