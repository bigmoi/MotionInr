# import pytorch_lightning as pl
# import torch; print("torch =", torch.__version__, "| cuda?", torch.cuda.is_available())
# print("pl =", pl.__version__)
#
#
import argparse
from omegaconf import OmegaConf
from lightning import Trainer

#择出cli与配置文件定义trainer中不同的参数，返回参数key
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser




if __name__ == '__main__':
    parser=get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()  # 对于addarguments中没有的参数放在unknown中

    if opt.resume:  # 实现加载checkpoint
        pass

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)  # 使用命令行未知参数覆盖配置文件中的参数，保证cli参数优先级最高

    # 合并后的总配置 config 中取出键 lightning 对应的子配置，并赋给 lightning_config，若空则赋值一个空的 OmegaConf 对象
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    #配置训练器
    trainer_config["accelerator"] = "cuda"
    trainer_config["strategy"] = "ddp"

    # 出现同名参数时，优先使用cli在参数覆盖.
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False


    trainer.fit()
    trainer.train()