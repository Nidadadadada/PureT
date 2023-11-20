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
class Generator(object):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        # 设置随机数种子
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
            """
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            """

        # 单机多卡
        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        # SCST标记
        self.rl_stage = False
        # 设置日志写入
        self.setup_logging()
        # 训练数据集
        self.setup_dataset()
        # 训练模型结构
        self.setup_network()
        # 模型验证
        self.val_evaler = Evaler(
            eval_ids=cfg.DATA_LOADER.VAL_ID,  # 图像id文件  './mscoco/txt/coco_val_image_id.txt'
            gv_feat=cfg.DATA_LOADER.VAL_GV_FEAT,
            att_feats=cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.VAL_ANNFILE
        )
        self.test_evaler = Evaler(
            eval_ids=cfg.DATA_LOADER.TEST_ID,  # 图像id文件  './mscoco/txt/coco_test_image_id.txt'
            gv_feat=cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats=cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.TEST_ANNFILE
        )
        self.scorer = Scorer()

    # 设置日志写入
    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        # 使用多卡训练时不输出日志
        if self.distributed and dist.get_rank() > 0:
            return

        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        # 模型构建
        generator = models.create(cfg.MODEL.TYPE)
        # print(generator)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.generator = torch.nn.parallel.DistributedDataParallel(
                generator.to(self.device),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False
            )
        else:
            self.generator = torch.nn.DataParallel(generator).cuda()

        # 如果resume > 0，则需要导入参数
        # 此处导入参数到CPU上？
        if self.args.resume > 0:
            self.generator.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                           map_location=lambda storage, loc: storage)
            )

        # 判断是否导入epoch
        self.load_epoch = -1
        self.load_iteration = -1
        if self.args.load_epoch:
            self.load_epoch = self.args.resume - 1  # 保存的resume名称从1计数
            # 4000是训练样本数量 unsupervised里面照片的数量
            self.load_iteration = int(self.args.resume * 4000 / cfg.TRAIN.BATCH_SIZE)

        # 训练优化器
        # load_iteration为scheduler中使用的last_epoch，
        # 用于简单粗略的恢复学习率，只对NoamOpt作用
        # 完整恢复optimizer，还是得保存checkpoint文件
        self.optim = Optimizer(self.generator, self.load_iteration)
        # 训练损失计算
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()

    # 训练数据集导入
    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT
        )

    # DataLoader
    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    # 模型验证
    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        # 验证集上测试结果
        val_res = self.val_evaler(self.generator, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        # 测试集上测试结果
        test_res = self.test_evaler(self.generator, 'test_' + str(epoch + 1))
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        # 返回模型路径：experiments/snapshot/{MODELNAME}_{epoch}.pth
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    # 保存模型
    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.generator.state_dict(), self.snapshot_path("GAN_model", epoch + 1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:, 0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs

    # 返回scheduled sampling概率
    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.generator.module.ss_prob = ss_prob

    # 训练数据显示
    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str + ', lr = ' + str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    def make_dis_dataloader(self,cap,dis,discriminator_tokenizer,discriminator_args):
        # 这里generator和discriminator应该保持一致，都是17
        max_seq_a_length = discriminator_args.max_seq_a_length
        dataset={}
        with open('MOCS/rewrite.tsv') as f:
            rewrite = f.readlines()
        len_generate_sentence = len(cap)
        cap.extend(rewrite)

        for i in range(len(cap)):
            caption = cap[i]
            # TODO: 这里用str还是int
            if i < len_generate_sentence:
                label = 1
            else:
                label = 0
            tokens = discriminator_tokenizer.tokenize(caption)
            if len(tokens) > max_seq_a_length - 1:
                tokens = tokens[:(max_seq_a_length - 1)]
            tokens = [discriminator_tokenizer.cls_token] + tokens
            seq_len = len(tokens)
            padding_len = max_seq_a_length - seq_len
            tokens = tokens + ([discriminator_tokenizer.pad_token] * padding_len)
            input_ids = discriminator_tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids,dtype = torch.long)
            caption_id = i
            dataset[caption_id] = {}
            dataset[caption_id]['input_ids'] = input_ids
            dataset[caption_id]['labels'] = torch.tensor([label])
            dataset[caption_id]['token_type_ids'] = torch.zeros_like(input_ids)
            dataset[caption_id]['attention_mask'] = torch.ones_like(input_ids)
        val = [list(item.values()) for item in dataset.values()]
        dataset = Discriminator_Dataset(val)
        shuffle = False

        sampler = dis.make_data_sampler(dataset,shuffle,distributed=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,num_workers=discriminator_args.num_workers, sampler = sampler,
            batch_size = len(dataset),
            pin_memory = True
        )
        return data_loader

    # 模型损失计算过程
    def forward(self, kwargs,dis,discriminator,discriminator_tokenizer,discriminator_args):

        # XE训练过程损失计算
        logits = self.generator(**kwargs)
        print('logits.requires_grad',logits.requires_grad)
        logit = torch.max(logits, -1)[1].data
        cap=[]

        for tensor in logit:
            a_cap = discriminator_tokenizer.decode(tensor.tolist(), skip_special_tokens=True)
            cap.append(a_cap)
        #这里要掺着正确的句子，不然模型啥也学不到
        data_loader = self.make_dis_dataloader(cap,dis,discriminator_tokenizer,discriminator_args)

        discriminator.eval()

        for step, batch in tqdm(enumerate(data_loader)):

            batch = tuple(t.to(discriminator_args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[3],
                'labels': batch[1]
            }

            outputs = discriminator(**inputs)
            loss, logits = outputs[:2]
        # print('outputs.requires_grad',outputs.requires_grad)
        print('loss.requires_grad',loss.requires_grad)

        return loss


    # 模型训练过程
    def generator_train(self,dis,discriminator,discriminator_tokenizer,discriminator_args):
        self.generator.train()
        # self.optim.zero_grad()
        iteration = self.load_iteration + 1
        # Epoch迭代
        for epoch in range(self.load_epoch + 1, cfg.SOLVER.MAX_EPOCH):
            if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            # 设置DataLoader
            self.setup_loader(epoch)

            running_loss = .0
            running_reward_baseline = .0
            # 每一个Epoch内部Iteration迭代
            with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.training_loader)) as pbar:
                for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(
                        self.training_loader):
                    # data_time.update(time.time() - start)
                    input_seq = input_seq.cuda()
                    target_seq = target_seq.cuda()
                    gv_feat = gv_feat.cuda()
                    att_feats = att_feats.cuda()
                    att_mask = att_mask.cuda()

                    kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
                    # 1、计算模型损失（XE训练 或 SCST训练）
                    loss = self.forward(kwargs,dis,discriminator,discriminator_tokenizer,discriminator_args)

                    # 2、梯度清零（清空过往梯度）
                    self.optim.zero_grad()
                    # 3、计算新梯度及梯度裁剪

                    # loss.requires_grad_(True)
                    loss.backward()  # 非混合精度训练

                    utils.clip_gradient(self.optim.optimizer, self.generator,
                                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                    # 4、权重更新
                    self.optim.step()  # 非混合精度训练
                    # 5、（XE）、优化器lr更新（用于XE训练），在SCST时不起作用
                    self.optim.scheduler_step('Iter')

                    running_loss += loss.item()
                    if not self.rl_stage:
                        pbar.set_postfix(
                            loss='%.2f' % (running_loss / (_ + 1))
                        )
                    else:
                        running_reward_baseline += loss_info['reward_baseline']
                        pbar.set_postfix(
                            {'loss/r_b': '%.2f/%.2f' % (running_loss / (_ + 1), running_reward_baseline / (_ + 1))}
                        )
                    pbar.update()
                    # print(str(self.optim.get_lr()))
                    iteration += 1

                    if self.distributed:
                        dist.barrier()

            # 每一个Epoch结束保存模型
            self.save_model(epoch)
            # 模型验证测试，返回的val仅用于SCST训练过程
            val = self.eval(epoch)
            # 4（SCST）、优化器lr更新（用于SCST训练），在XE训练时不起作用
            # 4 (XE)、优化器lr更新，当使用Step学习率策略时作用
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()

def generator_parse_args(folder=None,local_rank=0,resume=-1,load_epoch=False):
    '''
    Parse input arguments
    '''
    generator_parser = argparse.ArgumentParser(description='Image Captioning')
    # generator_parser.add_argument('--folder', dest='folder', type=str, default='experiments_PureT/PureT_XE')
    # generator_parser.add_argument("--local_rank", type=int, default=0)
    # generator_parser.add_argument("--resume", type=int, default=11)
    # generator_parser.add_argument("--load_epoch", action='store_true')
    # if len(sys.argv) == 1:
    #     generator_parser.print_help()
    #     sys.exit(1)
    # generator_args = generator_parser.parse_args()
    generator_parser.set_defaults(folder='experiments_PureT/PureT_XE',local_rank=0,resume=11,load_epoch=False)
    generator_args = generator_parser.parse_args()
    return generator_args

class Discriminator_Dataset(Dataset):
    def __init__(self, data):
        self.alldata = data

    def __getitem__(self, index):
        return self.alldata[index]

    def __len__(self):
        return len(self.alldata)

class Discriminator_DC(object):

    def build_dataset(self,args, tokenizer):
        wrong_sentences = pd.read_csv('wrong_sentences.csv')
        max_seq_a_length = args.max_seq_a_length
        dataset = {}

        for i in range(len(wrong_sentences)):
            caption, label = wrong_sentences['content'][i], wrong_sentences['label'][i]
            tokens = tokenizer.tokenize(caption)

            if len(tokens) > max_seq_a_length - 1:
                tokens = tokens[:(max_seq_a_length - 1)]

            tokens = [tokenizer.cls_token] + tokens
            seq_len = len(tokens)
            padding_len = max_seq_a_length - seq_len
            tokens = tokens + ([tokenizer.pad_token] * padding_len)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)

            caption_id = i
            dataset[caption_id] = {}
            dataset[caption_id]['input_ids'] = input_ids
            dataset[caption_id]['labels'] = torch.tensor([label])
            # token_type_ids值为0或1区分token属于第一句还是第二句 和 segment_ids是一个东西
            dataset[caption_id]['token_type_ids'] = torch.zeros_like(input_ids)
            dataset[caption_id]['attention_mask'] = torch.ones_like(input_ids)

        value = [list(item.values()) for item in dataset.values()]
        x_trval, x_test = train_test_split(value[:], test_size=0.2, random_state=42)
        x_train, x_eval = train_test_split(x_trval, test_size=0.5, random_state=42)

        # if is_train:
        #     return Discriminator_Dataset(x_train)
        # else:
        #     return Discriminator_Dataset
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
        logger.info("DC Train with {} images per GPU.".format(captions_per_gpu))
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

        logger.info("***** DC Running training *****")
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                    args.per_gpu_train_batch_size * self.get_world_size() * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step, global_loss, global_acc = 0, 0.0, 0.0
        model.zero_grad()

        writer = SummaryWriter("Tensorboard_Files/DC")  # 定义logs文件位置
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
                                                                                                     optimizer.param_groups[
                                                                                                         0]["lr"], loss,
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
            for step, batch in tqdm(enumerate(test_dataloader)):

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

def discriminator_dc_parse_args():

    discriminator_parser = argparse.ArgumentParser()
    discriminator_parser.set_defaults(model_name_or_path='./bert-base-uncased',output_dir='output/bert/dc',loss_type='sfmx',max_seq_a_length=17,do_train=True,\
                                      do_test=True,do_eval=True,drop_out=0.1,tie_weight=False,freeze_embedding=False,label_smoothing=0,drop_worst_ratio=0,\
                                      drop_worst_after=0,per_gpu_train_batch_size=64,per_gpu_eval_batch_size=64,gradient_accumulation_steps=1,\
                                      learning_rate=3e-5,weight_decay=0.05,adam_epsilon=1e-8,max_grad_norm=1.0,warmup_steps=0,scheduler='linear',num_workers=4,\
                                      num_train_epochs=40,max_steps=-1,logging_steps=16,save_steps=-1,evaluate_during_training=True,no_cuda=False,local_rank=0,\
                                      seed=88,eval_model_dir='',tie_weights=False)

    discriminator_args = discriminator_parser.parse_args()
    return discriminator_args

def main():
    training_epoch=2
    #创建generator
    my_folder='experiments_PureT/PureT_XE'
    my_local_rank=1
    my_resume=0
    generator_args = generator_parse_args(my_folder,my_local_rank,my_resume)
    print(generator_args)
    if generator_args.folder is not None:
        cfg_from_file(os.path.join(generator_args.folder, 'config.yml'))
    cfg.ROOT_DIR = generator_args.folder
    generator = Generator(generator_args)
    #创建discriminator
    discriminator_args = discriminator_dc_parse_args()
    dis = Discriminator_DC()
    global logger
    local_rank = dis.ensure_init_process_group(local_rank=discriminator_args.local_rank)

    discriminator_args.local_rank = local_rank
    discriminator_args.num_gpus = dis.get_world_size()
    discriminator_args.distributed = discriminator_args.num_gpus > 1
    discriminator_args.device = torch.device('cuda')
    dis.synchronize()

    output_dir = discriminator_args.output_dir
    mkdir(output_dir)

    logger = setup_logger("train", output_dir, discriminator_args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", discriminator_args.device, discriminator_args.num_gpus)
    set_seed(discriminator_args.seed, discriminator_args.num_gpus)
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    discriminator_config = BertConfig.from_pretrained('./bert-base-uncased', num_labels=2)

    discriminator_tokenizer = tokenizer_class.from_pretrained('./bert-base-uncased', do_lower_case=True)
    discriminator_config.hidden_dropout_prob = discriminator_args.drop_out
    discriminator_config.loss_type = discriminator_args.loss_type
    discriminator_config.tie_weights = discriminator_args.tie_weights
    discriminator_config.freeze_embedding = discriminator_args.freeze_embedding
    discriminator_config.label_smoothing = discriminator_args.label_smoothing
    discriminator_config.drop_worst_ratio = discriminator_args.drop_worst_ratio
    discriminator_config.drop_worst_after = discriminator_args.drop_worst_after
    discriminator = model_class.from_pretrained('./bert-base-uncased/pytorch_model.bin', config=discriminator_config)
    discriminator.to(discriminator_args.device)

    for i in range(training_epoch):
        discriminator_train_dataloader, discriminator_val_dataloader,discriminator_test_dataloader= dis.make_data_loader(discriminator_args, discriminator_tokenizer,discriminator_args.distributed)
        # discriminator_val_dataloader = dis.make_data_loader(discriminator_args, discriminator_tokenizer, discriminator_args.distributed, is_train=False)
        last_checkpoint = dis.train(discriminator_args, discriminator_train_dataloader, discriminator_val_dataloader, discriminator, discriminator_tokenizer)
        # discriminator_test_dataloader = dis.make_data_loader(discriminator_args, discriminator_tokenizer, discriminator_args.distributed, is_train=False)

        dis.test(discriminator_args, discriminator_test_dataloader, discriminator)

        generator.generator_train(dis,discriminator,discriminator_tokenizer,discriminator_args)


if __name__ == '__main__':
    main()






