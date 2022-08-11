# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import logging
import os
import random
import timeit
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    BartConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # BartForConditionalGeneration
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from kat.utils import pretty_print_cecap
from kat.data_lino import KATDataset, KATDataCollator
from kat.modeling_lino import BartForConditionalGeneration
from transformers import BartForConditionalGeneration as OriginBart
from kat.modeling_basic_BART import BartForConditionalGeneration as LinoOriginBart
from dial.metrics import dialogue_evaluation
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

torch.set_printoptions(precision=4)

logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from kat.data_lino import Example  # has to be defined for torch.load in data_lino.py

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': f'{total_num:,}', 'Trainable': f'{trainable_num:,}'}

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def grad_status(model: nn.Module):
    return (par.requires_grad for par in model.parameters())

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def load_kno_mlm_weight(model:BartForConditionalGeneration, mlmmodel_path:str):
    mlmmodel:OriginBart = OriginBart.from_pretrained(mlmmodel_path)
    logger.info("init knowledge encoder and knowledge attention with another Transformer")
    mlm_state_dict = mlmmodel.model.encoder.state_dict()
    knowl_encoder_dict = model.model.knowl_encoder.state_dict()
    for name in knowl_encoder_dict:
        knowl_encoder_dict[name].copy_(mlm_state_dict[name])
    mlm_state_dict = mlmmodel.model.decoder.state_dict()
    knowl_decoder_dict = model.model.decoder.state_dict()
    for name in knowl_decoder_dict:
        if 'knowl_attn' in name:
            pretrain_name = name.replace('knowl_attn', 'encoder_attn')
            knowl_decoder_dict[name].copy_(mlm_state_dict[pretrain_name])

def read_ks_ordering(checkpoint):
    from ks.json_file_check import remove_repetition

    file_name = os.path.join(checkpoint, 'pred_record.json')
    with open(file_name, 'r') as f:
        pred_record = json.load(f)
        pred_record = remove_repetition(pred_record)

    return pred_record

def train(args, train_dataset, model: BartForConditionalGeneration, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(filename_suffix=args.task)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=KATDataCollator(tokenizer, args, is_train=True))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.fix_encoder:
        assert_all_frozen(model.model.encoder)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #     os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
       
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         global_step = int(checkpoint_suffix)
    #         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from epoch %d", epochs_trained)
    #         logger.info("  Continuing training from global step %d", global_step)
    #         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    #     except ValueError:
    #         logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            # kno_input_ids = batch["kno_input_ids"].to(args.device)
            # kno_attention_mask = batch["kno_attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            ks_labels = batch['ks_labels'].to(args.device) if args.do_ks else None
            inputs = {
                # "input_ids": (input_ids, kno_input_ids),
                "input_ids": input_ids,
                # "attention_mask": (attention_mask, kno_attention_mask),
                "attention_mask": attention_mask,
                "labels": labels,
                "return_dict": True,
                'ks_labels': ks_labels
                # "reduction": "mean",
            }
            loss = model(**inputs).loss

            # if args.n_gpu > 1: # always take mean
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
               
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # print("lino's test 03.31 average cand kno num: %.2f" % (1.0*sum(args.cand_kno_num)/len(args.cand_kno_num)))
    # print("lino's test 03.31 average real kno num: %.2f" % (1.0*sum(args.real_kno_num)/len(args.real_kno_num)))

    return global_step, tr_loss / global_step

def evaluate(args, model:BartForConditionalGeneration, tokenizer, prefix=""):
    dataset = KATDataset(
        tokenizer,
        type_path=args.eval_prefix,
        data_dir=args.data_dir,
        n_obs=-1,
        # n_obs = 200,
        max_target_length=args.max_target_length,
        max_source_length=args.max_source_length,
        prefix=model.config.prefix or "",
        cecap=False,
        ts_dir=args.ts_dir
    )
    if args.eval_ks == False:
        model.falsify_do_ks()
    if args.sort_kno == 'ks_pred_record':
        # checkpoint = 'ks/checkpoints/ks_roberta_cecap/checkpoint-63000'
        checkpoint = args.ks_checkpoint
        pred_record = read_ks_ordering(checkpoint)

        new_pred_record = {}
        for key, value in pred_record.items():
            new_pred_record[key.replace(' ', '')] = value  # pred_record's key and dialogue src are a bit different.
        args.pred_record = new_pred_record
    if args.sort_kno == 'tf-idf':
        args.recall_result = {'score':{1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0,
                                       6:0.0, 7:0.0, 8:0.0, 9:0.0, 10:0.0},
                              'total_num': 0}


    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    golds = dataset.read_targets()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=KATDataCollator(tokenizer, args))

    # multi-gpu evaluate
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    write_exids = []
    last_example_id = 0
    write_histories = []
    write_knowledges = []
    write_labels = []
    ppls = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        # kno_input_ids = batch["kno_input_ids"].to(args.device)
        # kno_attention_mask = batch["kno_attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        inputs = {
            "input_ids": input_ids,
            # "input_ids": (input_ids, kno_input_ids),
            "attention_mask": attention_mask,
            # "attention_mask": (attention_mask, kno_attention_mask),
            "labels": labels,
            "return_dict": True,
            # "reduction": "none",
        }
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            mask = labels == tokenizer.pad_token_id
            # (bsz, len)
            loss = model(**inputs).loss.view(batch_size, -1).masked_fill(mask, 0)
            ppl = (loss.sum(dim=1) / (1 - mask.float()).sum(dim=1)).exp()
            ppls.extend(ppl.tolist())

        seqs = model.generate(
            input_ids=input_ids,
            max_length=args.max_target_length,
            use_cache=True,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            do_sample=args.do_sample, early_stopping=True,
            top_k=args.top_k, top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
        )
        # seqs = model.generate(
        #     input_ids=(input_ids, kno_input_ids), max_length=args.max_target_length, use_cache=True,
        #     attention_mask=(attention_mask, kno_attention_mask), num_beams=args.num_beams,
        #     do_sample=args.do_sample, early_stopping=True,
        #     top_k=args.top_k,
        #     no_repeat_ngram_size=args.no_repeat_ngram_size,
        #     repetition_penalty=args.repetition_penalty,
        #     length_penalty=args.length_penalty,
        # )
        seqs_for_eval = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        seqs = tokenizer.batch_decode(seqs, skip_special_tokens=False)

        # seqs = tokenizer.batch_decode(seqs, skip_special_tokens=False)
        preds.extend(seqs_for_eval)

        hh = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # hh = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        for bi in range(len(hh)):
            hh[bi] = hh[bi].replace('<pad>', '')
        write_histories.extend(hh)

        # for i in range(kno_input_ids.size(0)):
        #     kk = tokenizer.batch_decode(kno_input_ids[i], skip_special_tokens=True)
        #     write_knowledges.append(kk)

        # kk = tokenizer.batch_decode(kno_input_ids, skip_special_tokens=True)  # 32 x 40 x 64

        ll = tokenizer.batch_decode(labels, skip_special_tokens=False)
        # ll = tokenizer.batch_decode(labels, skip_special_tokens=False)
        for bi in range(len(ll)):
            ll[bi] = ll[bi].replace('<pad>', '')
        write_labels.extend(ll)

        write_exids.extend([i for i in range(last_example_id, last_example_id+batch_size)] )
        last_example_id += batch_size

        # break  # 리노 수정

    evalTime = timeit.default_timer() - start_time
    assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}."
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    output_prediction_file = outdir / "predictions_{}_{}_{}.txt".format(args.task, args.eval_prefix, prefix)
    output_result_file = outdir / "results_{}_{}_{}.json".format(args.task, args.eval_prefix, prefix)
    logger.info(f" Writing predictions into {str(output_prediction_file)}")
    logger.info(f" Writing results into {str(output_result_file)}")

    # write_result_to_json(preds, write_histories, write_knowledges, write_labels, outdir)
    write_result_to_json(preds, write_exids, write_histories, write_labels, outdir)
    # print('exit by Lino')
    # exit(10)
    if args.sort_kno == 'tf-idf':
        print(args.recall_result)

    with output_prediction_file.open('w') as f:
        for line in preds:
            f.write(line.strip() + "\n")
    results = dialogue_evaluation(preds, golds)
    results['ppl'] = sum(ppls) / len(ppls)
    with output_result_file.open('w') as f:
        json.dump(results, f, indent=4)
    return results

def write_result_to_json(preds, wexids, wh, wl, outdir):
    '''
    Write inference sentence with its history, knowledge candidates, original label
    to the json file.
    '''
    objects = []
    for p, exid, h, l in zip(preds, wexids, wh, wl):
        object = {
            '0. example id': exid,
            '1. history': h,
            '2. label': l,
            '3. model prediction': p,
        }
        objects.append(object)

    with open(outdir / "phl_result.json", 'w') as json_file:
        json.dump(objects, json_file, indent=4, sort_keys=True)

# def write_result_to_json(preds, wh, wk, wl, outdir):
#     '''
#     Write inference sentence with its history, knowledge candidates, original label
#     to the json file.
#     '''
#     objects = []
#     for p, h, k, l in zip(preds, wh, wk, wl):
#         object = {
#             '1. history': h,
#             '2. knowledge': k,
#             '3. label': l,
#             '4. model prediction': p
#         }
#         objects.append(object)
#
#     with open(outdir / "phkl_result.json", 'w') as json_file:
#         json.dump(objects, json_file, indent=4, sort_keys=True)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--kno_mlm_model_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )
    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_prefix",
        default="train",
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--eval_prefix",
        default="test",
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_knowl_length",
        default=10,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=40,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_num_kno",
        default=40,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_kno_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--n_obs",
        default=-1,
        type=int,
        help="The number of training examples",
    )
    parser.add_argument(
        "--cecap",
        default=False,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--do_ks",
        default=False,
        action='store_true',
        required=False,
        help="indicate whether to do the knowleddge selection task."
    )
    parser.add_argument(
        "--pred_record",
        default=None,
        required=False,
        help="ks prediction record dictionary"
    )
    parser.add_argument(
        "--ks_checkpoint",
        default=None,
        required=False,
        help="ks prediction record dictionary"
    )
    parser.add_argument(
        "--recall_result",
        default=None,
        required=False,
        help="recall result for tf-idf"
    )
    parser.add_argument(
        "--eval_ks",
        default=False,
        action='store_true',
        required=False,
        help="indicate whether to evaluate on the knowleddge selection task."
    )
    parser.add_argument(
        "--ts_dir",
        default=None,
        type=str,
        help="If not None, it indicates that the model will use the ts result from this checkpoint.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--is_kat_tslf", action="store_true", help="Set this flag if you are using the kat-tslf model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--top_p",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--repetition_penalty",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--length_penalty",
        default=None,
        type=float,
    )
    parser.add_argument("--shuffle_knowledge", action="store_true", help="Whether not to shuffle knowledge pool.")
    parser.add_argument("--include_true_kno", action="store_true", help="Whether not to always include the true knowledge in the input.")
    parser.add_argument("--only_true_kno", action="store_true", help="Whether to include only the true knowledge from the candidates to the model input.")
    parser.add_argument("--with_wow", action="store_true",
                        help="For the case of cecap training, include the wow data as well.")
    parser.add_argument("--sort_kno", type=str, default=False, help="Whether to sort the knowledge by given method")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--fix_encoder", action="store_true")
    parser.add_argument("--init_kno_encoder", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--cand_kno_num", type=list, default=[])  # lino's test 03.31
    parser.add_argument("--real_kno_num", type=list, default=[])  # lino's test 03.31
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config:BartConfig = BartConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # model = BartForConditionalGeneration.from_pretrained(
    # model = OriginBart.from_pretrained(
    if args.do_ks != False:
        cdict = config.to_dict()
        cdict['do_ks'] = True
        cdict['kno_num'] = args.max_num_kno
        config.update(cdict)
        # config.update_from_string("do_ks=true,kno_num=args.max_num_kno")

    model = LinoOriginBart.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # config.encoder_layers = 2
    # config.decoder_layers = 2
    # model = BartForConditionalGeneration(config)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
            apex.amp.register_float_function(torch, 'sigmoid')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        if args.init_kno_encoder:
            assert args.kno_mlm_model_path is not None
            logger.info("init knowledge encoder and knowledge attention with another Transformer")
            model.model.load_addi_weights()
        if args.fix_encoder:
            logger.info("fix utterance and utterance encoder")
            freeze_params(model.model.encoder)
            freeze_params(model.model.knowl_encoder)
            assert_all_frozen(model.model.encoder)
            assert_all_frozen(model.model.knowl_encoder)
        logger.info(f"#Params: {get_parameter_number(model)}")

        if args.cecap != False:
            args.cecap = [cecap.strip() for cecap in args.cecap.split(',')]  # make it a list of cecap names

        train_dataset = KATDataset(
            tokenizer,
            type_path=args.train_prefix,
            data_dir=args.data_dir,
            # n_obs=-1,
            # n_obs=287367,
            n_obs=args.n_obs,
            max_target_length=args.max_target_length,
            max_source_length=args.max_source_length,
            prefix=model.config.prefix or "",
            cecap=args.cecap, # 'cecap_wikipedia', 'cecap_reddit'
            with_wow=args.with_wow,
            # max_num_kno=args.max_num_kno,
        )
        pretty_print_cecap(train_dataset, args.data_dir, args.cecap)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_cls.from_pretrained(args.output_dir)  # , force_download=True)
        # tokenizer = token_cls.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # if args.do_train:
        logger.info("Loading checkpoints saved during training for evaluation")
        if args.output_dir in ['bart', 'bart_woft']: # there exists no trained model. Only using the baseline model from huggingface.

            checkpoints = [args.model_name_or_path]
        else:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        # else:
        #     logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        #     checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            # model = BartForConditionalGeneration.from_pretrained(checkpoint)  # , force_download=True)
            # model = OriginBart.from_pretrained(checkpoint)  # , force_download=True)
            model = LinoOriginBart.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)
            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    main()

