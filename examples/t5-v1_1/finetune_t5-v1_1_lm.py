import time
import random
import os
import csv
import sys

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import bmtrain as bmt

from model_center import get_args
from model_center.model import T5
from model_center.tokenizer import T5Tokenizer
from model_center.dataset.t5dataset_lm import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader


def get_tokenizer(args):
    tokenizer = T5Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = T5.from_pretrained(args.model_config)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 100)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size, device):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_encoder_length, args.max_decoder_length, device)
        bmt.print_rank(split, "max_enc_size:", dataset[split].max_enc_size)
        bmt.print_rank(split, "max_dec_size:", dataset[split].max_dec_size)
    verbalizer = torch.LongTensor(DATASET[dataset_name].get_verbalizer(tokenizer)).to(device)
    return dataset, verbalizer


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer, device):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")

    # print_inspect(model, '*')

    bmt.print_rank(verbalizer)

    for epoch in range(20):
        dataloader = {
            "train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=dataset["train"].collate),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=False, collate_fn=dataset["dev"].collate),
        }

        model.train()
        for it, data in enumerate(dataloader['train']):
            # bmt.print_rank("input_ids.size", data["input_ids"].size())
            # bmt.print_rank("input", data["input_ids"][2].cpu().tolist())
            # # bmt.print_rank("input str", tokenizer.decode(data["input_ids"][0].cpu().tolist()))
            # bmt.print_rank("decoder input", data["decoder_input_ids"][2].cpu().tolist())
            # # bmt.print_rank("decoder input str", tokenizer.decode(data["decoder_input_ids"][0].cpu().tolist()))
            # bmt.print_rank("length", data["length"][2])
            # bmt.print_rank("decoder length", data["decoder_length"][2])
            # bmt.print_rank("targets", data["targets"][2])
            # # bmt.print_rank("targets str", tokenizer.decode(data["targets"][0].cpu().tolist()))
            # bmt.print_rank("index", data["index"][2])

            optimizer.zero_grad()

            logits = model(**data, return_logits=True)
            
            bs, seq_len, vocab_size = logits.size()
            
            # bmt.print_rank(logits.size())
            logits = logits.view(-1, vocab_size)
            # bmt.print_rank(logits.size())

            targets = data["targets"].view(-1)
            
            loss = loss_func(logits, targets)

            # bmt.print_rank(loss)
            loss = loss.view(bs, seq_len)
            # bmt.print_rank(loss)
            loss = loss * data["loss_mask"]
            # bmt.print_rank(loss)
            loss = torch.sum(loss, dim=-1) / torch.sum(data["loss_mask"], dim=-1)
            # bmt.print_rank(loss)
            loss = torch.mean(loss, dim=0)
            # bmt.print_rank(loss)
            # exit(0)
            
            # logits = logits.index_select(dim=-1, index=verbalizer)
            # logits = logits[torch.where(data["index"]==1)]

            # loss = loss_func(logits, data["targets"])
            global_loss = bmt.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    grad_norm,
                )
            )
            # if it % args.inspect_iters == 0: print_inspect(model, "*")
            # if args.save != None and it % args.save_iters == 0:
            #     bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % it)))

        model.eval()
        with torch.no_grad():
            for split in ['dev']:
                pd = []
                gt = []
                for it, data in enumerate(dataloader[split]):

                    logits = model(**data, return_logits=True)
                    logits = logits.index_select(dim=-1, index=verbalizer)
                    logits = logits[torch.where(data["index"]==1)]
                    preds = logits.argmax(dim=-1)
                    
                    # print(preds)
                    # print(data["labels"])
                
                    pd.extend(preds.cpu().tolist())
                    gt.extend(data["labels"].cpu().tolist())

                    bmt.print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} |".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                        )
                    )
                pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
                gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()
                bmt.print_rank(pd)
                bmt.print_rank(gt)
                
                bmt.print_rank(f"{split} epoch {epoch}:")
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    bmt.print_rank(f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    f1 = f1_score(gt, pd, average="macro")
                    bmt.print_rank(f"Average F1: {f1*100:.2f}")


def main():
    args = initialize()
    device = torch.cuda.current_device()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(), device
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer, device)

if __name__ == "__main__":
    main()
