# main.py
import os, argparse, datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from Utils.utils import instantiate_from_config  # 若你有自己的 instantiate_from_config，沿用即可
import torch
# def parse_args():
#     parser = argparse.ArgumentParser("Minimal Lightning Runner")
#     parser.add_argument("--config", "-c",type=str,default='cfgs/h36m/dct_ddim_predenoiser.yml', help="Path to YAML config")
#     parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.ckpt) to resume")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--logdir", type=str, default="./logs")
#     # 允许用 --override key1=a,b key2.sub=3 覆盖配置
#     parser.add_argument("--override", nargs="*", default=[], help="Override config keys: k=v")
#     return parser.parse_args()
#
# def load_config(cfg_path, overrides):
#     cfg = OmegaConf.load(cfg_path)
#     if overrides:
#         # 支持 "a.b=3" 点式覆盖
#         dot = OmegaConf.from_dotlist(overrides)
#         cfg = OmegaConf.merge(cfg, dot)
#     return cfg
#
# def make_logger(cfg_lightning, log_root, run_name):
#     # 若 YAML 中定义了 logger，就按配置实例化；否则用最简单的 TensorBoard
#     if cfg_lightning and "logger" in cfg_lightning:
#         return instantiate_from_config(cfg_lightning.logger)
#     return TensorBoardLogger(save_dir=log_root, name=run_name)
#
# def main():
#     args = parse_args()
#     cfg = load_config(args.config, args.override)
#
#     # 基础目录与随机种子
#     run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     run_name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0]) + f"_{run_time}"
#     log_root = os.path.abspath(args.logdir)
#     os.makedirs(log_root, exist_ok=True)
#     seed_everything(args.seed, workers=True)
#
#     # 1) 实例化模型与数据模块（都来自 YAML）
#     # 期望 YAML 结构含有:
#     # model: {target: "...Class", params:{...}}
#     # data:  {target: "...DataModuleClass", params:{...}}
#     model = instantiate_from_config(cfg.model)
#     datamodule = instantiate_from_config(cfg.data)
#
#     # 2) 日志与回调
#     lightning_cfg = cfg.get("lightning", {})
#     logger = make_logger(lightning_cfg, log_root, run_name)
#
#     ckpt_dir = os.path.join(logger.save_dir, logger.name if hasattr(logger, "name") else os.path.join(os.getcwd(),f"checkpoints/{cfg.configname}"), "checkpoints")
#     os.makedirs(ckpt_dir, exist_ok=True)
#
#     # 必需回调：保存最后一个 & （如有 monitor）保存最优
#     ckpt_cb = ModelCheckpoint(
#         dirpath=ckpt_dir,
#         filename="{epoch:04d}",
#         save_last=True,
#         save_top_k=1 if getattr(model, "monitor", None) else 0,
#         monitor=getattr(model, "monitor", None),
#         mode=getattr(model, "monitor_mode", "min"),
#         auto_insert_metric_name=True,
#     )
#     lr_cb = LearningRateMonitor(logging_interval="step")
#
#     # 3) Trainer（若 YAML 给了 lightning.trainer 就用，没有就走简洁默认）
#     trainer_kwargs = {"logger": logger, "callbacks": [ckpt_cb, lr_cb], "default_root_dir": log_root}
#     if "trainer" in lightning_cfg:
#         trainer_kwargs.update(OmegaConf.to_container(lightning_cfg.trainer, resolve=True))
#
#     trainer = Trainer(**trainer_kwargs)
#
#     # 记录配置到 logger（WandB/TensorBoard 都安全）
#     if hasattr(logger, "log_hyperparams"):
#         logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
#
#     # 4) 训练（可选恢复）
#     ckpt_path = args.resume if args.resume else None
#     trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
#
#     # 可选：验证/测试（如果 DataModule 定义了）
#     # trainer.validate(model, datamodule=datamodule)
#     # trainer.test(model, datamodule=datamodule)
#
# if __name__ == "__main__":
#     main()
#



def parse_args():
    parser = argparse.ArgumentParser("Minimal Lightning Runner")
    parser.add_argument("--config", "-c", type=str,
                        default="cfgs/h36m/dct_ddim_predenoiser.yml",
                        help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.ckpt) to resume")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in YAML (seed_everything)")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config keys: k=v (dotlist)")
    return parser.parse_args()

def load_config(cfg_path, overrides):
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        dot = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, dot)
    return cfg

def make_logger(cfg):
    # cfg.logger 是 {target: ..., params: {...}} 的结构
    if "logger" in cfg and cfg.logger:
        try:
            return instantiate_from_config(cfg.logger)
        except Exception as e:
            print(f"[warn] instantiate logger from cfg failed: {e}. Fallback to TensorBoard.")
    # 兜底：TensorBoard
    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = cfg.get("configname", "run") + f"_{run_time}"
    return TensorBoardLogger(save_dir=os.path.abspath("./logs"), name=name)

def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)

    # ----------------- 基础设置 -----------------
    # 优先用 CLI 的 --seed，否则用 YAML 的 seed_everything，否则用 42
    seed = args.seed if args.seed is not None else cfg.get("seed_everything", 42)
    seed_everything(int(seed), workers=True)

    # 可选：响应 PyTorch 提示，开启 Tensor Cores 更激进的 matmul 精度
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 日志根目录
    log_root = os.path.abspath(args.logdir)
    os.makedirs(log_root, exist_ok=True)

    # ----------------- 实例化 model / data -----------------
    model = instantiate_from_config(cfg.model)   # 需要 {target, params}
    datamodule = instantiate_from_config(cfg.data)

    # ----------------- Logger & Callbacks -----------------
    logger = make_logger(cfg)

    # checkpoint 存在一个稳定路径（不依赖 logger 内部属性）
    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = cfg.get("configname", os.path.splitext(os.path.basename(args.config))[0]) + f"_{run_time}"
    ckpt_dir = os.path.join(log_root, run_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:04d}",
        save_last=True,
        save_top_k=0,            # 如需监控指标可改成 1 并设置 monitor/mode
        auto_insert_metric_name=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # ----------------- 组装 Trainer -----------------
    trainer_kwargs = {
        "logger": logger,
        "callbacks": [ckpt_cb, lr_cb],
        "default_root_dir": log_root,
    }

    # 关键修正：从 cfg.trainer（顶层）读入 Trainer 配置
    if "trainer" in cfg and cfg.trainer:
        trainer_kwargs.update(OmegaConf.to_container(cfg.trainer, resolve=True))

    trainer = Trainer(**trainer_kwargs)

    # 记录完整配置（W&B / TB 都 OK）
    try:
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print(f"[warn] logger.log_hyperparams failed: {e}")

    # ----------------- 训练（可恢复） -----------------
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
