import json
import os
import sys
import pprint
import random
import time
# import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm,trange
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter

import losses
import models
import datasets
import lib.utils as utils
import pandas as pd
import os.path as op
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
from utils.misc import (mkdir, set_seed)
from utils.logger import setup_logger
from sklearn.model_selection import train_test_split

from transformers.pytorch_transformers import BertTokenizer, BertConfig,BertForSequenceClassification
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
# """


class Discriminator_Dataset(Dataset):
    def __init__(self, data):
        self.alldata = data

    def __getitem__(self, index):
        return self.alldata[index]

    def __len__(self):
        return len(self.alldata)

class Discriminator_MC(object):

    def build_dataset(self,args, tokenizer):
        with open('MOCS/supervised_att_feats.json') as f:
            supervised_att_feats = json.load(f)
        # TODO 转成 dict形式
        with open('MOCS/new_finalresult.tsv')  as f:
            supervised_caption = f.readlines()[:1000]
        wrong_pair={}
        true_pair={}

        for id in supervised_att_feats:
            if id not in wrong_pair:
                wrong_pair[id]=[]
                wrong_pair[id].append(supervised_caption[id])
                randint = np.random.randint(581930,582929)
                wrong_pair[id].append(supervised_att_feats[randint])

                true_pair[id] = []
                true_pair[id].append(supervised_caption[id])
                true_pair[id].append(supervised_att_feats[id])

        wrong_sentences = pd.read_csv('wrong_sentences.csv')
        max_seq_a_length = args.max_seq_a_length
        dataset = {}

        for id in list(wrong_pair.keys()):
            caption,att, label = wrong_pair[id][0], wrong_pair[id][1],1
            tokens = tokenizer.tokenize(caption)

            if len(tokens) > max_seq_a_length - 1:
                tokens = tokens[:(max_seq_a_length - 1)]

            tokens = [tokenizer.cls_token] + tokens
            seq_len = len(tokens)
            padding_len = max_seq_a_length - seq_len
            tokens = tokens + ([tokenizer.pad_token] * padding_len)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)

            caption_id = id
            dataset[caption_id] = {}
            dataset[caption_id]['input_ids'] = input_ids
            dataset[caption_id]['att'] = att
            dataset[caption_id]['labels'] = torch.tensor([label])
            # token_type_ids值为0或1区分token属于第一句还是第二句 和 segment_ids是一个东西
            dataset[caption_id]['token_type_ids'] = torch.zeros_like(input_ids)
            dataset[caption_id]['token_type_ids'] = torch.cat((dataset[caption_id]['token_type_ids'],torch.tensor([1])),-1)
            # 这个attention只跟句子有关系吗
            dataset[caption_id]['attention_mask'] = torch.ones_like(input_ids)

        for id in list(true_pair.keys()):
            caption,att, label = wrong_pair[id][0], wrong_pair[id][1],0
            tokens = tokenizer.tokenize(caption)

            if len(tokens) > max_seq_a_length - 1:
                tokens = tokens[:(max_seq_a_length - 1)]

            tokens = [tokenizer.cls_token] + tokens
            seq_len = len(tokens)
            padding_len = max_seq_a_length - seq_len
            tokens = tokens + ([tokenizer.pad_token] * padding_len)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)

            caption_id = id
            dataset[caption_id] = {}
            dataset[caption_id]['input_ids'] = input_ids
            dataset[caption_id]['att'] = att
            dataset[caption_id]['labels'] = torch.tensor([label])
            # token_type_ids值为0或1区分token属于第一句还是第二句 和 segment_ids是一个东西
            dataset[caption_id]['token_type_ids'] = torch.zeros_like(input_ids)
            dataset[caption_id]['token_type_ids'] = torch.cat((dataset[caption_id]['token_type_ids'],torch.tensor([1])),-1)
            # 这个attention只跟句子有关系吗
            dataset[caption_id]['attention_mask'] = torch.ones_like(input_ids)


        value = [list(item.values()) for item in dataset.values()]
        x_trval, x_test = train_test_split(value[:], test_size=0.2, random_state=42)
        x_train, x_eval = train_test_split(x_trval, test_size=0.5, random_state=42)

        return Discriminator_Dataset(x_train),Discriminator_Dataset(x_eval),Discriminator_Dataset(x_test)

    def make_data_sampler(self,dataset, shuffle, distributed):
        if distributed:
            return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler

    def make_data_loader(self,args, tokenizer, is_distributed=False):
        train_dataset,eval_dataset,test_dataset = self.build_dataset(args, tokenizer)

        shuffle = True
        captions_per_gpu = args.per_gpu_train_batch_size
        captions_per_batch = captions_per_gpu * self.get_world_size()
        iters_per_batch = len(train_dataset) // captions_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("MC Train with {} images per GPU.".format(captions_per_gpu))
        logger.info("Total batch size {}".format(captions_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        train_sampler=self.make_data_sampler(train_dataset, shuffle, is_distributed)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.num_workers, sampler=train_sampler,
            batch_size=captions_per_gpu,
            pin_memory=True,
        )

        shuffle = False
        captions_per_gpu = args.per_gpu_eval_batch_size
        eval_sampler = self.make_data_sampler(eval_dataset, shuffle, is_distributed)
        test_sampler = self.make_data_sampler(test_dataset, shuffle, is_distributed)
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset, num_workers=args.num_workers, sampler=eval_sampler,
            batch_size=captions_per_gpu,
            pin_memory=True,
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, num_workers=args.num_workers, sampler=test_sampler,
            batch_size=captions_per_gpu,
            pin_memory=True,
        )
        return train_data_loader,eval_data_loader,test_data_loader

    def save_checkpoint(self,model, tokenizer, args, epoch, iteration, num_trial=10):
        checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
            epoch, iteration))
        if not self.is_main_process():
            return checkpoint_dir
        mkdir(checkpoint_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        for i in range(num_trial):
            try:
                model_to_save.save_pretrained(checkpoint_dir)
                torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info("Save checkpoint to {}".format(checkpoint_dir))
                break
            except:
                pass
        else:
            logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
        return checkpoint_dir

    def compute_score_with_logits(logits, labels):
        logits = torch.max(logits, -1)[1].data  # argmax
        scores = logits == labels
        return scores

    # 精度计算
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self,args, train_dataloader, val_dataloader, model, tokenizer):
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                                                       args.gradient_accumulation_steps) + 1
        else:
            # 改变参数的次数
            t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                      * args.num_train_epochs

        # Prepare optimizer and scheduler(linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not \
                any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if \
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.scheduler == "constant":
            scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
        elif args.scheduler == "linear":
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        else:
            raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

        logger.info("***** MC Running training *****")
        # logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
        #             args.per_gpu_train_batch_size * self.get_world_size() * args.gradient_accumulation_steps)
        # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)

        global_step, global_loss, global_acc = 0, 0.0, 0.0
        model.zero_grad()

        writer = SummaryWriter("Tensorboard_Files/MC")  # 定义logs文件位置
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

        for epoch in train_iterator:
            # 这种会生成序号
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                model.train()

                inputs = {'input_ids': batch[0], 'token_type_ids': batch[2],
                          'attention_mask': batch[3], 'labels': batch[1]
                          }
                labels = batch[1]
                outputs = model(**inputs)
                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # 反向梯度信息
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                global_loss += loss.item()
                # global_acc += batch_acc
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    if global_step % args.logging_steps == 0:
                        # 这应该写一个评估
                        logger.info(
                                "Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), ".format(epoch, global_step,
                                                                                                     optimizer.param_groups[0]["lr"], loss,
                                                                                                     global_loss / global_step))
                        writer.add_scalar("Loss", global_loss / global_step, epoch)

                    if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                            global_step == t_total:
                        checkpoint_dir = self.save_checkpoint(model, tokenizer, args, epoch, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        writer.close()
        return checkpoint_dir

    def test(self,args, test_dataloader, model):
        world_size = self.get_world_size()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        time_meter = 0

        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_dataloader),position=0):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[3],
                    'labels': batch[1]
                }

                tic = time.time()
                outputs = model(**inputs)
                loss, logits = outputs[:2]
                time_meter += time.time() - tic
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                labels = batch[1].to('cpu').numpy()
                total_eval_accuracy += self.flat_accuracy(logits, labels)
            print('type(total_eval_loss)',type(total_eval_loss))
            avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
            print("Accuracy: %.4f" % (avg_val_accuracy))
            print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
            print("-------------------------------")
            # writer.close()  #关闭
            logger.info(
                "Inference model computing time: {} seconds per batch,Accuracy:{:.4f},Average testing loss{:.4f}".format(
                    time_meter / (step + 1), avg_val_accuracy, total_eval_loss / len(test_dataloader)))

        if world_size > 1:
            torch.distributed.barrier()
        return total_eval_loss / len(test_dataloader)

    def get_world_size(self):
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def synchronize(self):
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()

    def ensure_init_process_group(self,local_rank=None, port=12345):
        # init with env
        world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        if world_size > 1 and not dist.is_initialized():
            assert local_rank is not None
            print("Init distributed training on local rank {}".format(local_rank))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl', init_method='env://'
            )
        return local_rank

def discriminator_mc_parse_args():

    discriminator_parser = argparse.ArgumentParser()
    discriminator_parser.set_defaults(model_name_or_path='./bert-base-uncased',output_dir='output/bert/mc',loss_type='sfmx',max_seq_a_length=17,do_train=True,\
                                      do_test=True,do_eval=True,drop_out=0.1,tie_weight=False,freeze_embedding=False,label_smoothing=0,drop_worst_ratio=0,\
                                      drop_worst_after=0,per_gpu_train_batch_size=64,per_gpu_eval_batch_size=64,gradient_accumulation_steps=1,\
                                      learning_rate=3e-5,weight_decay=0.05,adam_epsilon=1e-8,max_grad_norm=1.0,warmup_steps=0,scheduler='linear',num_workers=4,\
                                      num_train_epochs=40,max_steps=-1,logging_steps=16,save_steps=-1,evaluate_during_training=True,no_cuda=False,local_rank=0,\
                                      seed=88,eval_model_dir='',tie_weights=False)

    discriminator_args = discriminator_parser.parse_args()
    return discriminator_args











