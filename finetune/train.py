# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import math
import os
import random
import time
from functools import partial
from pprint import pprint

import numpy as np
import paddle
# from datasets import load_dataset
from paddlenlp.datasets import load_dataset
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from tqdm import tqdm

from utils import compute_metrics, main_process_first, save_ckpt, print_args

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import (
    LinearDecayWithWarmup,
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from data_loader import convert_example_human_activity, read_file, convert_example_news, truncate_news

from paddlenlp.transformers.opt.modeling import OPTForCausalLM
import jieba


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        type=str,
        help="Path to pre-trained model. ",
    )
    parser.add_argument("--train_file", type=str, required=False, default="data/train.json", help="Train data path.")
    parser.add_argument("--eval_file", type=str, required=False, default="data/test.json", help="Eval data path.")
    parser.add_argument(
        "--save_dir",
        default="output",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=0,
        type=int,
        help="The minimum total sequence length for target text when generating. ",
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--epoch",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_interval", type=float, default=1, help="Save checkpoint every X epoch.")
    parser.add_argument("--eval_interval", type=float, default=1, help="Evaluate model performance every X epoch.")
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion",
    )
    parser.add_argument(
        "--warmup_proportion", default=0.1, type=float, help="Linear warmup proportion over total steps."
    )
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override epoch.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.",
    )
    parser.add_argument("--use_amp", default=False, type=strtobool, help="Enable mixed precision training.")
    parser.add_argument("--scale_loss", default=2 ** 15, type=float, help="The value of scale_loss for fp16.")
    parser.add_argument("--use_SSTIA", action="store_true", help="Whether to use SSTIA.")
    parser.add_argument("--mix_ratio", default=0, type=float, help="Mixture ratio for TSDASG synthetic input.")
    parser.add_argument("--do_lower_case", default=1, type=int, choices=[0, 1])
    parser.add_argument("--metric_weights", type=float, default=[1.0, 0.9, 1, 1], nargs="*")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Checkpoint to warm start from.")
    parser.add_argument("--eval_checkpoint", type=int, choices=[0, 1], default=1)
    parser.add_argument("--task", type=str, default="human_activity", choices=["human_activity", "news"])
    parser.add_argument("--use_activity_name", default=True, type=strtobool,
                        help="is use activity name for source(for human-activity)")
    parser.add_argument("--expansion_coef", default=1.4, type=float,
                        help="max_source_length_of_char=max_source_length*expansion_coef")
    parser.add_argument("--head2tail", default=[3, 1], type=float, nargs='*',
                        help="ratio of body head-to-tail truncation(for news-summary)")
    parser.add_argument("--user_dict", type=str, default='./data/vocab/user_dict_for_jieba.txt')
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, data_loader, tokenizer, min_target_length, max_target_length, use_SSTIA):
    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    if use_SSTIA:
        model.use_SSTIA = False
    for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
        labels = batch.pop("labels").numpy()
        preds = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            min_length=min_target_length,
            max_length=max_target_length,
            use_cache=True,
        )[0]
        all_preds.extend(
            tokenizer.batch_decode(preds.numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    metrics = compute_metrics(all_preds, all_labels)
    model.train()
    if use_SSTIA:
        model.use_SSTIA = True
    return metrics


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    # load model and tokenizer
    model = PegasusForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = PegasusChineseTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.use_SSTIA:
        model.use_SSTIA = True
        model.mix_ratio = args.mix_ratio

    # load-from-checkpoint
    is_from_checkpoint = False
    if args.init_checkpoint:
        try:
            model_state = paddle.load(os.path.join(args.init_checkpoint, "model_state.pdparams"))
            model.set_state_dict(model_state)
            tokenizer = AutoTokenizer.from_pretrained(args.init_checkpoint, do_lower_case=args.do_lower_case)
            jieba.load_userdict(args.user_dict)
            print(f"custom vocab: {args.user_dict} loaded by jieba.")
            is_from_checkpoint = True
            print(f"checkpoint loaded from {args.init_checkpoint}.")
        except Exception as e:
            print(e.__str__())
            print(f"\ncheckpoint load failed from {args.init_checkpoint}.")
    if not args.do_lower_case:
        assert is_from_checkpoint, "if do_lower_case==False, it should load model from checkpoint"

    # data-loader
    train_set = load_dataset(read_file, file=args.train_file, lazy=False)
    dev_set = load_dataset(read_file, file=args.eval_file, lazy=False)
    remove_columns = ["content", "title"]
    if args.task == "human_activity":
        trans_func = partial(
            convert_example_human_activity,
            text_column="content",
            summary_column="title",
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            expansion_coef=args.expansion_coef,
            use_activity_name=args.use_activity_name
        )
    if args.task == "news":
        trans_func = partial(
            convert_example_news,
            summary_column="summary",
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            truncate_func=truncate_news,
            expansion_coef=args.expansion_coef,
            ration_head2tail=args.head2tail
        )
    with main_process_first(desc="train dataset map pre-processing"):
        # train_set = train_set.map(trans_func, batched=False, load_from_cache_file=True, remove_columns=remove_columns)
        train_set = train_set.map(trans_func, lazy=True)
    with main_process_first(desc="dev dataset map pre-processing"):
        # dev_set = dev_set.map(trans_func, batched=False, load_from_cache_file=True, remove_columns=remove_columns)
        dev_set = dev_set.map(trans_func, lazy=True)

    batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_batch_sampler = DistributedBatchSampler(train_set, batch_size=args.train_batch_size, shuffle=True)

    train_data_loader = DataLoader(
        dataset=train_set, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    )

    dev_batch_sampler = BatchSampler(dev_set, batch_size=args.eval_batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_set, batch_sampler=dev_batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    )
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.epoch
        num_train_epochs = args.epoch

    save_steps = int(len(train_data_loader) * args.save_interval)
    eval_steps = int(len(train_data_loader) * args.eval_interval)
    logger.debug(f"num_training_steps: {num_training_steps}, num_train_epochs: {num_train_epochs}.")
    logger.debug(f"eval every {args.eval_interval} epoch, {eval_steps} steps.")
    logger.debug(f"save every {args.save_interval} epoch, {save_steps} steps.")

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )
    # debug
    """
    optimizer = Lion(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.99,
        parameters=model.parameters(),
        weight_decay=args.weight_decay
    )
    """

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    best_metrics = (0, 0, 0, 0)
    best_metrics_avg = 0
    metric_weights = args.metric_weights
    if is_from_checkpoint and args.eval_checkpoint:
        metrics = evaluate(
            model, dev_data_loader, tokenizer, args.min_target_length, args.max_target_length, args.use_SSTIA
        )
        metrics_avg = sum(np.array(metrics) * metric_weights / sum(metric_weights))
        logger.info("init checkpoint: rouge1: %.4f, rouge2: %.4f, rougel: %.4f, bleu: %.4f, avg: %.4f\n" % (
            metrics[0], metrics[1], metrics[2], metrics[3], metrics_avg))
        best_metrics = metrics
        best_metrics_avg = metrics_avg

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            with paddle.amp.auto_cast(args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]):
                lm_logits, new_cache, loss = model(**batch)
            if args.use_amp:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (
                        global_step,
                        num_training_steps,
                        epoch,
                        step,
                        paddle.distributed.get_rank(),
                        loss,
                        optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()
            if global_step % save_steps == 0 or global_step == num_training_steps:
                save_ckpt(model, tokenizer, args.save_dir, global_step)
                logger.info("Saved step {} model.\n".format(global_step))
            if global_step % eval_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                metrics = evaluate(
                    model, dev_data_loader, tokenizer, args.min_target_length, args.max_target_length, args.use_SSTIA
                )
                metrics_avg = sum(np.array(metrics) * metric_weights / sum(metric_weights))
                logger.info(
                    "epoch %d - step %05d: rouge1: %.4f, rouge2: %.4f, rougel: %.4f, bleu: %.4f, avg: %.4f" % (
                        epoch, step, metrics[0], metrics[1], metrics[2], metrics[3], metrics_avg))
                logger.info("           best_last: "
                            "rouge1: %.4f, rouge2: %.4f, rougel: %.4f, bleu: %.4f, avg: %.4f"
                            % (best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3], best_metrics_avg))
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0 and metrics_avg > best_metrics_avg:
                    best_metrics_avg = metrics_avg
                    best_metrics = metrics
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    # Need better way to get inner model of DataParallel
                    save_ckpt(model, tokenizer, args.save_dir, "best")
                    logger.info(f"Saved step {global_step} model as model best.\n")
            if global_step > num_training_steps:
                return


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    do_train(args)
