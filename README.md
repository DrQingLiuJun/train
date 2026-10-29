# 路侧目标检测训练项目

目录：`/home/user/carla-projects/train`；服务器：`user@172.16.4.81`。

旧 A100 虚拟机已收回。本项目从本机保存的脚本、数据集和模型重建；**没有恢复最近那次 YOLO26m 训练的 last.pt/best.pt、优化器或训练轮数**。models/baseline 中的0428.pt、best1.pt、last1.pt是更早的三个模型，不代表最近训练结果。

## 启动训练

```bash
cd /home/user/carla-projects/train
./start.sh
```

默认从官方YOLO26m预训练权重开始：768像素、batch16、最多250轮、连续40轮无改善早停。配置在`config/train.yaml`。安装过程不自动启动正式训练。GPU与CARLA及激光雷达服务共用；当前默认批量为新服务器重新设置，未设置显存比例上限。需要独占显卡时先由操作者停止其他业务。

## 查看、停止和恢复

```bash
./status.sh --watch
# 查看批次进度：
tail -n 30 -f runs/main/train.log
# 停止本项目训练：
./stop.sh
# 恢复中断的main训练（需要包含优化器和有效epoch的完整检查点）：
./start.sh --mode resume --name main
```

Ctrl+C退出watch或tail不影响训练。每轮验证完成后更新mAP50、mAP50–95、召回率、精确率。后台训练不会因SSH断开而停止。

输出：`runs/main/weights/best.pt`、`last.pt`、`resume.pt`。resume.pt是每轮覆盖的完整检查点，保留优化器供中断恢复；只有这一份滚动副本，不产生逐轮历史快照。已正常结束/早停的训练与中断恢复不同，若要额外训练应明确选择微调并使用新名称，不伪造旧优化器状态。

从旧模型继续微调：

```bash
./start.sh --mode finetune --model models/baseline/0428.pt --name finetune_0428
```

微调使用lr0=0.0002、warmup_epochs=0、warmup_bias_lr=0，避免成熟权重被高偏置暖启动学习率扰动。完整resume缺失优化器时直接报错，不会静默改为微调。微调轮数是新阶段轮数，不冒充之前累计轮数。旧模型保持原有架构，不会因文件名变成YOLO26m。

## 评估

```bash
.venv/bin/python scripts/evaluate.py --model runs/main/weights/best.pt
.venv/bin/python scripts/evaluate.py --model models/baseline/0428.pt
```

使用同一验证集，768像素、conf0.001、IoU0.45，结果写入reports/evaluation.json。单次推理/小样本测试不能证明新训练精度。

## 数据集

来源为用户指定dataset.zip。仅提取其中正式的images/train、images/val、labels/train、labels/val；不把new_dataset候选材料、旧脚本、压缩包内模型混入正式训练。

传输后核对整个压缩包SHA256，并将每张正式图片SHA256与旧项目本地保存的审计记录比对。类别为person/car/cyclist/truck。保留旧项目的数据准备规则：缺失标注不当作负样本；去重、跳过非法标注行；训练集排除非法/高度跨类别重叠图片，并裁剪越界训练框；验证集保留原分组及清理后的有效标注，歧义样本会在审计中披露。空标注文件可作为背景图保留。报告在reports/dataset-audit.json。

清理仅发生在新服务器的训练副本，本机原压缩包不变。有效图片清单是dataset/train.txt和val.txt，data.yaml已改为本机绝对路径。数据准备完成后删除服务器临时压缩包和安装包，仅保留可用数据、模型、脚本和审计报告。

## 环境

独立`.venv`，不修改CARLA或rslidar_ws虚拟环境。PyTorch2.8.0+cu128 / torchvision0.23.0来自[PyTorch官方版本索引](https://pytorch.org/get-started/previous-versions/)，Ultralytics固定8.4.190，NumPy1.26.4/OpenCV4.11.0。requirements-lock.txt记录实际安装版本。实际CUDA、模型加载、小样本前向/反向与完整检查点测试见reports/environment.json。


## GitHub 归档与当前续训（2026-10-29）

仓库保存代码、配置、依赖清单、数据审计报告、训练参数及 results.csv。数据集图片、标注、train.txt/val.txt、模型权重、虚拟环境和大日志均保留在服务器，不属于这次 GitHub 代码归档。克隆仓库本身不能还原完整训练环境；dataset/data.yaml 和历史参数使用服务器绝对路径。数据准备脚本还依赖服务器上的 dataset.zip 和 .reference-images.json，后者未包含在仓库。

当前 main 已完成 84 轮并正常结束，目标上限为 250 轮、patience=40；这不是训练进程意外中断。最终最佳模型验证日志约为 Precision=0.890、Recall=0.840、mAP50=0.893、mAP50-95=0.685；训练状态报告中的指标是第84轮指标，与最佳模型不同。

在原服务器从本次最佳权重继续微调，示例为新阶段最多 100 轮（仍会早停）：

```bash
cd /home/user/carla-projects/train
./start.sh --mode finetune --model runs/main/weights/best.pt --name finetune_main_01 --epochs 100 --batch 32
./status.sh --watch
```

该命令重新建立优化器，以 lr0=0.0002 开始新阶段；新阶段轮数从 1 开始，历史 main 结果保留。若同名阶段已有结果，请改用新名称。启动后也可用 `tail -f runs/finetune_main_01/train.log` 查看日志。退出查看不会停止后台训练。

只有需要恢复尚未正常结束的训练时，才使用完整状态恢复：

```bash
./start.sh --mode resume --name finetune_main_01
```

完整恢复使用对应阶段 weights/resume.pt（模型、优化器、轮次），而非已移除优化器的 best.pt/last.pt。原 main 的 resume.pt 保存至零基 epoch=83，即完成84轮，文件约219 MB；直接恢复 main 会沿用原先的训练设置和早停状态，不等于重新追加100轮。请勿把 `--mode resume --epochs ...` 当作可靠的追加轮数方式，实际轮数由 Ultralytics 的恢复逻辑处理。

本次上传没有启动训练，也没有更改服务器上的原始项目文件。
