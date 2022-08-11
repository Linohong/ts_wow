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
import re
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
import bisect

from transformers import (
    WEIGHTS_NAME,
    BartConfig,
    AutoTokenizer,
    # BartForConditionalGeneration
    AutoModelForSequenceClassification,
    AutoConfig,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from ts.ts_data_lino import KATDataset, KATDataCollator, ts_Example
from ts.ts_utils import possible_topics, write_ts_result_to_file
from ts.ts_modeling import tsRobertaModel
from kat.modeling_lino import BartForConditionalGeneration
from transformers import BartForConditionalGeneration as OriginBart
# from kat.modeling_basic_BART import BartForConditionalGeneration as LinoOriginBart
from dial.metrics import dialogue_ks_evaluation

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

def process_pred_record(original_id, score, dialog_history, topic_cand, topic_sent, label, pred_record):
    if original_id not in pred_record:
        pred_record[original_id] = []
    pred_record[original_id].append({'score':score, 'dialog_history': dialog_history,
                                     'topic_cand': topic_cand, 'topic_sent': topic_sent,
                                     'label': label == 0})

    # if src in pred_record:
    #     if pred_record[src]['score'] < float(score):
    #         pred_record[src] = {'score': float(score),
    #                             'response': response,
    #                             'label': int(label) == 0}
    # else:
    #     pred_record[src] = {'score': float(score),
    #                         'response': response,
    #                         'label': int(label) == 0}


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

def train(args, train_dataset, model: BartForConditionalGeneration, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir, filename_suffix=args.task)

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
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

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

    tr_loss, logging_loss = 0.0, 0.0
    tot_tr_loss, tot_logging_loss = 0.0, 0.0
    ts_tr_loss, ts_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    epoch_num = 1
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
            labels = batch["labels"].to(args.device)
            if args.do_tot:
                tot_labels = batch['tot_labels'].to(args.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "tot_labels": tot_labels,
                "return_dict": True,
                # "reduction": "mean",
            }
            if args.do_tot:
                ts_loss, tot_loss = model(**inputs).loss
                ts_loss = ts_loss.mean()
                tot_loss = tot_loss.mean()
                loss = ts_loss + tot_loss
            else:
                loss = model(**inputs).loss


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            if args.do_tot:
                tot_tr_loss += tot_loss.item()
                ts_tr_loss += ts_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
               
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 25 == 0:
                    if args.do_tot:
                        logger.info('MEAN TS_LOSS FOR step at %s: %s' % (global_step, ts_loss.item()))
                        logger.info('MEAN TOT_LOSS FOR step at %s: %s' % (global_step, tot_loss.item()))
                        logger.info("Learning rate at %s: %s" % (global_step, scheduler.get_lr()[0]))

                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("ts_loss", (ts_tr_loss - ts_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("tot_loss", (tot_tr_loss - tot_logging_loss) / args.logging_steps, global_step)
                    else:
                        logger.info('MEAN LOSS FOR step at %s: %s' % (global_step, loss.item()))
                        logger.info("Learning rate at %s: %s" % (global_step, scheduler.get_lr()[0]))

                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

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
                    tot_logging_loss = tot_tr_loss
                    ts_logging_loss = ts_tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    to = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    save_model(args, model.module if hasattr(model, "module") else model,
                               tokenizer, to, optimizer.state_dict(), scheduler.state_dict())
                    logger.info("Saving model checkpoint to %s", to)
                    logger.info("Saving optimizer and scheduler states to %s", to)
                    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    # model_to_save = model.module if hasattr(model, "module") else model
                    # model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # save the model after the epoch
        save_model(args, model.module if hasattr(model, "module") else model,
                   tokenizer, os.path.join(args.output_dir, "checkpoint-{}".format(global_step)),
                   optimizer.state_dict(), scheduler.state_dict(), epoch=epoch_num)

        # epoch done at this line.

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # print("lino's test 03.31 average cand kno num: %.2f" % (1.0*sum(args.cand_kno_num)/len(args.cand_kno_num)))
    # print("lino's test 03.31 average real kno num: %.2f" % (1.0*sum(args.real_kno_num)/len(args.real_kno_num)))

    return global_step, tr_loss / global_step

def evaluate(args, model:BartForConditionalGeneration, tokenizer, prefix="", checkpoint=None):
    dataset = KATDataset(
        tokenizer,
        type_path=args.eval_prefix,
        data_dir=args.data_dir,
        n_obs=-1,
        max_target_length=args.max_target_length,
        max_source_length=args.max_source_length,
        prefix=model.config.prefix or "",
        cecap=False,
    )
    dataset = transpose_examples_to_ts(dataset, is_train=False, SEP=tokenizer.sep_token, do_tot=args.do_tot)  # dataset has 'is_train' info.

    # if args.eval_ks == False:
    #     model.falsify_do_ks()

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # golds = dataset.read_targets()
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
    pred_record = {}  # key: src sentence ---> {'score': 0.9674, 'repsonse': "____", 'label': True/False}

    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        original_ids = batch['original_ids']
        if args.do_tot != False:
            tot_labels = batch['tot_labels'].to(args.device)
        else:
            tot_labels = None
        # eval_ids = batch['eval_ids']

        # inputs = {
        #     "input_ids": input_ids,
        #     # "input_ids": (input_ids, kno_input_ids),
        #     "attention_mask": attention_mask,
        #     # "attention_mask": (attention_mask, kno_attention_mask),
        #     "labels": labels,
        #     "return_dict": True,
        #     # "reduction": "none",
        # }

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tot_labels": tot_labels,
            "return_dict": True,
            # "reduction": "mean",
        }
        if args.do_tot == False:
            inputs.pop('tot_labels')
            
        batch_size = input_ids.shape[0]

        # TODO: 2022.04.12 여기서부터 해야함. model output 살펴보고
        # example 들을 dialogue 단위별로 모아주고, 결과값 도출하는 코드를 작성해야한다.
        with torch.no_grad():
            mask = labels == tokenizer.pad_token_id
            # (bsz, len)
            output = model(**inputs)

        for i, logit in enumerate(output.logits):
            decoded_sent = tokenizer.decode(input_ids[i]).replace('<pad>', '')
            if tokenizer.sep_token == '</s>':
                splited_input_sent = decoded_sent.split(tokenizer.sep_token+tokenizer.sep_token)
                src, topic, topic_sents = splited_input_sent[0], splited_input_sent[1], splited_input_sent[2]
            else:
                splited_input_sent = decoded_sent.split(tokenizer.sep_token+' '+tokenizer.sep_token)
                topic_sents = splited_input_sent[-1].strip().replace(tokenizer.pad_token, '')
                src, topic = splited_input_sent[0].split(tokenizer.sep_token)
                src = src.strip().replace(tokenizer.cls_token, '').strip()
                topic = topic.strip()

            process_pred_record(original_ids[i], -float(logit[0]), src, topic, topic_sents, int(labels[i]), pred_record)

    for key, info in pred_record.items():
        sorted_list = sorted(info, key=lambda d: d['score'])
        # sorted_list.reverse()
        pred_record[key] = sorted_list

    evalTime = timeit.default_timer() - start_time
    # assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}."
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    output_result_file = Path(checkpoint) / "ts_results.json"
    logger.info(f" Writing results into {str(output_result_file)}")

    results = dialogue_ks_evaluation(pred_record)
    with output_result_file.open('w') as f:
        json.dump(results, f, indent=4)

    detail_result_file = Path(checkpoint) / "ts_detail_results.json"
    logger.info(f" Writing detailed results into {str(detail_result_file)}")
    write_ts_result_to_file(detail_result_file, pred_record, true_to_the_top=True)

    return results, pred_record

def write_result_to_json(preds, wh, wl, outdir):
    '''
    Write inference sentence with its history, knowledge candidates, original label
    to the json file.
    '''
    objects = []
    for p, h, l in zip(preds, wh, wl):
        object = {
            '1. history': h,
            '2. label': l,
            '3. model prediction': p
        }
        objects.append(object)

    with open(outdir / "phl_result.json", 'w') as json_file:
        json.dump(objects, json_file, indent=4, sort_keys=True)

def transpose_examples_to_ts(train_dataset, is_train=True, num_false_cases=4, SEP='</s>', true_topic_for_kno_sent=False, do_tot=False):
    # The first entry of 'knowl' attribute is the label knowledge.
    new_examples = []
    with open('./ts/topic_to_document.json') as f:
        topic_to_document = json.load(f)

    do_tot_dataset_stats = {'true':0, 'false':0}

    if is_train:
        for original_id, e in enumerate(train_dataset):
            # if original_id == 10:
            #     break
            # true case
            if e.knowl_topic[0] != 'no_passages_used': # e.knowl_topic[0] 은 KAT-TSLF 데이터가 정렬한 데이터의 첫 번째 Knowledge. 따라서, 정답 knowledge sentence.
                try:
                    doc = topic_to_document[e.knowl_topic[0].lower().strip()]
                except KeyError:
                    doc = [kno for i, kno in enumerate(e.knowl) if e.knowl_topic[i] == e.knowl_topic[0]]  # 아쉬운대로 example 내의 문장들이라도 붙임

                # transition of the topic information
                tot_label = None
                if do_tot:
                    search = e.knowl_topic[0].lower()
                    search = re.sub(r'\([^)]*\)', '', search).strip()

                    context = e.src.lower().split('</s>')
                    # if search in context[-1] and search not in context[0]:
                    if search in context[-1]:
                        tot_label = True
                        do_tot_dataset_stats['true'] += 1
                    else:
                        tot_label = False
                        do_tot_dataset_stats['false'] += 1

                new_examples.append(
                    ts_Example(
                        src=e.src.lower(),
                        cand_topic=e.knowl_topic[0].lower(),  # true topic
                        topic_sents=SEP.join([d.lower() for d in doc]),
                        label=True,
                        is_train=is_train,
                        tot_label=tot_label,
                    )
                )


            other_topics = set(e.knowl_topic)
            other_topics = other_topics - set(['no_passages_used']) - set([e.knowl_topic[0]])

            # calculate true case possible topics and add examples accordingly.
            if e.knowl_topic[0] != 'no_passages_used' and true_topic_for_kno_sent != True:  # 유사도 비슷한 topic 들도 true 로 간주
                poss_topics = possible_topics(e.knowl_topic[0], list(other_topics))
                for possible_topic in poss_topics:
                    try:
                        doc = topic_to_document[possible_topic.lower().strip()]
                    except KeyError:
                        doc = [kno for i, kno in enumerate(e.knowl) if e.knowl_topic[i] == possible_topic]  # 아쉬운대로 example 내의 문장들이라도 붙임
                    new_examples.append(
                        ts_Example(
                            src=e.src.lower(),
                            cand_topic=possible_topic.lower(),  # true topic
                            topic_sents=SEP.join([d.lower() for d in doc]),
                            label=True,
                            is_train=is_train,
                        )
                )
            else:
                poss_topics = []

            unlikely_topics = other_topics - set(poss_topics)

            if len(unlikely_topics) == 0: # if there exists no negative cases from the current example
                rand_topics = random.sample(topic_to_document.keys(), k=10)
                unlikely_topics = list(set(rand_topics) - set(possible_topics(e.knowl_topic[0], rand_topics)))
                unlikely_topics = unlikely_topics[:1] # max 1 negative samples

            # false cases
            # selected_cases = list(range(1,len(e.knowl)))
            # if is_train and selected_cases!=[]:
            #     selected_cases = random.sample(selected_cases, k=min(num_false_cases, len(selected_cases)))
            unlikely_topics = random.sample(unlikely_topics, k=(len(unlikely_topics)//2)+1)  # 일단 절반만
            for neg_topic in list(unlikely_topics):
                do_tot_dataset_stats['false'] += 1

                try:
                    doc = topic_to_document[neg_topic.lower().strip()]
                except KeyError:
                    doc = [kno for i, kno in enumerate(e.knowl) if e.knowl_topic[i] == neg_topic]  # 아쉬운대로 example 내의 문장들이라도 붙임
                new_examples.append(
                    ts_Example(
                        src=e.src.lower(),
                        cand_topic=neg_topic.lower(),  # false topic
                        topic_sents=SEP.join([d.lower() for d in doc]),
                        label=False,
                        is_train=is_train,
                        tot_label=False if do_tot else None
                    )
                )
    else:  # is_train == False
        for original_id, e in enumerate(train_dataset):
            # if original_id == 10:
            #     break

            # true case
            if e.knowl_topic[0] != 'no_passages_used': # difficult cases
                try:
                    doc = topic_to_document[e.knowl_topic[0].lower().strip()]
                except KeyError:
                    doc = [kno for i, kno in enumerate(e.knowl) if e.knowl_topic[i] == e.knowl_topic[0]]  # 아쉬운대로 example 내의 문장들이라도 붙임

                tot_label = False
                if do_tot:
                    search = e.knowl_topic[0].lower()
                    search = re.sub(r'\([^)]*\)', '', search).strip()

                    context = e.src.lower().split('</s>')
                    # if search in context[-1] and search not in context[0]:
                    if search in context[-1]:
                        tot_label = True
                        do_tot_dataset_stats['true'] += 1
                    else:
                        tot_label = False
                        do_tot_dataset_stats['false'] += 1
                new_examples.append(
                    ts_Example(
                        src=e.src,
                        cand_topic=e.knowl_topic[0],  # true topic
                        topic_sents=SEP.join(doc),
                        label=True,
                        is_train=False,
                        original_id=original_id,
                        tot_label=tot_label,
                    )
                )

            # negative cases --- 일단, 정답 Knowledge 가 포함되어 있지 않은 topic 들은 모두 false 라고 함
            other_topics = set(e.knowl_topic)
            other_topics = other_topics - set(['no_passages_used']) - set([e.knowl_topic[0]])
            for other_topic in other_topics:
                try:
                    doc = topic_to_document[other_topic.lower().strip()]
                except KeyError:
                    doc = [kno for i, kno in enumerate(e.knowl) if e.knowl_topic[i] == other_topic]  # 아쉬운대로 example 내의 문장들이라도 붙임
                new_examples.append(
                    ts_Example(
                        src=e.src,
                        cand_topic=other_topic,  # true topic
                        topic_sents=SEP.join(doc),
                        label=False,
                        is_train=False,
                        original_id=original_id,
                        tot_label=False,
                    )
                )
                do_tot_dataset_stats['false'] += 1

    print('현재는 합쳐서입니다 (%s)' % ('TRAIN_DATA' if is_train else 'EVALUATION_DATA'))
    src_len = 0.0
    topic_sents_len = 0.0
    true_num = 0
    false_num = 0.0
    for e in new_examples:
        src_len += len(e.src.split(' '))
        topic_sents_len += len(e.topic_sents.split(' '))
        if e.label == True:
            true_num += 1
        else:
            false_num += 1

    print('[ *** AVERAGES STATS *** ] ')
    print("DO TOT dataset stats: %s" % do_tot_dataset_stats)
    print('source word nums : %f' % (src_len/len(new_examples)))
    print('cand word nums : %f' % (topic_sents_len/len(new_examples)))
    print('true example num : %f' % (true_num))
    print('false example num : %f' % (false_num))

    return new_examples

def write_pred_record(checkpoint, pred_record):
    file_name = os.path.join(checkpoint, 'pred_record.json')
    with open(file_name, 'w') as ksf:
        json.dump(pred_record, ksf, indent=4)

def save_model(args, model, tokenizer, checkpoint, optimizer_state_dict, scheduler_state_dict, epoch=None):
    # Save the trained model and the tokenizer
    if epoch==None:
        to = checkpoint
    else:
        to = checkpoint + ('_%s'%epoch)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", to)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(to)
        tokenizer.save_pretrained(to)
        torch.save(optimizer_state_dict, os.path.join(to, "optimizer.pt"))
        torch.save(scheduler_state_dict, os.path.join(to, "scheduler.pt"))
        # config.save_pretrained(to)

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
        "--eval_ks",
        default=False,
        action='store_true',
        required=False,
        help="indicate whether to evaluate on the knowleddge selection task."
    )
    parser.add_argument("--do_tot", action="store_true", help="Whether to add 'transition of the topic' loss to the model.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
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
    parser.add_argument("--true_topic_for_kno_sent", action="store_true", help="Set TRUE value to the topic only if the true knowledge sentence is inside that topic")
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

        # 04.06 lino's test.
        # device = torch.device('cpu')

        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        # 04.06 lino's test.
        # args.n_gpu = 0
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

    config: AutoConfig = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    cdict = config.to_dict()
    cdict['num_labels'] = 2
    cdict['problem_type'] = "single_label_classification"
    config.update(cdict)


    if args.do_tot:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = tsRobertaModel(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

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
            n_obs=-1,
            # n_obs=287367,
            max_target_length=args.max_target_length,
            max_source_length=args.max_source_length,
            prefix=model.config.prefix or "",
            cecap=args.cecap, # 'cecap_wikipedia', 'cecap_reddit'
            # max_num_kno=args.max_num_kno,
        )
        train_dataset = transpose_examples_to_ts(train_dataset, SEP=tokenizer.sep_token, true_topic_for_kno_sent=args.true_topic_for_kno_sent, do_tot=args.do_tot)
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

            # (2022.04.20)
            # remove_checkpoints = ['ks/checkpoints/ks_roberta_cecap_74092/checkpoint-18000',
            #                       'ks/checkpoints/ks_roberta_cecap_74092/checkpoint-27000',
            #                       'ks/checkpoints/ks_roberta_cecap_74092/checkpoint-9000']
            # for rc in remove_checkpoints:
            #     ri = checkpoints.index(rc)
            #     checkpoints.pop(ri)


        logger.info("Evaluate the following checkpoints: %s", checkpoints)



        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else checkpoint,
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            if args.do_tot:
                model = tsRobertaModel.from_pretrained(checkpoint)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result, pred_record = evaluate(args, model, tokenizer, prefix=global_step, checkpoint=checkpoint)

            write_pred_record(checkpoint, pred_record)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            print('*** CURRENT RESULT FOR BELOW ***')
            print(checkpoint)
            print(result)
            print()
            results.update(result)

    logger.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    main()

