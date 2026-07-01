import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf
import h5py

import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
import pathlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def find_ckpt(path):
    p = pathlib.Path(path)
    ret = p.rglob("*.ckpt")
    last = None
    for item in ret:
        last = item
    return str(last)

def get_avg_results(results):
    avg_result = defaultdict(lambda: [0.0,0])
    for x in results:
        for k in x.keys():
            s = avg_result[k]
            s[0] += x[k]
            s[1] += 1
    for k in avg_result.keys():
        avg_result[k][0] /= avg_result[k][1]   
    return avg_result

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # if not is_dist_avail_and_initialized():
        #     return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
# --- add these small helpers near the top (below imports) ---

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    m = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=1, keepdims=True)

def _standardize_binary_io(output, target):
    """
    Returns:
      y_prob: [N] float in [0,1] (positive-class probability)
      y_true: [N] int in {0,1}
    Accepts:
      output: [N,2] (logits or probs), [N,1]/[N] (logits or probs)
      target: [N], [N,1], or one-hot [N,2]
    """
    y_out = _to_numpy(output)
    y_true = _to_numpy(target)

    # targets -> [N] {0,1}
    if y_true.ndim == 2 and y_true.shape[1] == 2:
        y_true = y_true.argmax(axis=1)
    elif y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = (y_true[:, 0] > 0.5).astype(np.int64)
    elif y_true.ndim == 1:
        y_true = y_true.astype(np.int64)
    else:
        raise ValueError(f"Unexpected target shape for binary: {y_true.shape}")

    # outputs -> prob [N] in [0,1]
    if y_out.ndim == 2 and y_out.shape[1] == 2:
        prob = _softmax_2d(y_out)[:, 1]
    elif y_out.ndim == 2 and y_out.shape[1] == 1:
        v = y_out[:, 0]
        prob = 1.0 / (1.0 + np.exp(-v)) if (v.min() < 0.0 or v.max() > 1.0) else v
    elif y_out.ndim == 1:
        v = y_out
        prob = 1.0 / (1.0 + np.exp(-v)) if (v.min() < 0.0 or v.max() > 1.0) else v
    else:
        raise ValueError(f"Unexpected output shape for binary: {y_out.shape}")

    return prob.astype(np.float32), y_true.astype(np.int64)

def _standardize_multiclass_io(output, target):
    """
    Returns:
      y_out: [N, C] array (logits or probs)
      y_true: [N] int labels
    """
    y_out = _to_numpy(output)
    y_true = _to_numpy(target)

    if y_true.ndim == 2:
        y_true = y_true.argmax(axis=1)
    elif y_true.ndim != 1:
        raise ValueError(f"Unexpected target shape for multiclass: {y_true.shape}")
    y_true = y_true.astype(np.int64)

    if y_out.ndim == 2:
        return y_out, y_true
    elif y_out.ndim == 1:
        num_classes = int(y_out.max()) + 1 if y_out.size > 0 else 1
        oh = np.full((y_out.shape[0], num_classes), -1e9, dtype=np.float32)
        oh[np.arange(y_out.shape[0]), y_out.astype(np.int64)] = 0.0
        return oh, y_true
    else:
        raise ValueError(f"Unexpected output shape for multiclass: {y_out.shape}")

def _normalize_metric_names(metrics, is_binary: bool):
    """
    Map user/legacy names to pyhealth names and remember how to map back.
    For binary: map any weighted/macro variants to 'f1' (pyhealth supports plain 'f1').
    """
    norm = []
    back = []  # original names to map results back
    for m in (metrics if isinstance(metrics, (list, tuple)) else [metrics]):
        orig = m
        ml = m.lower()
        # common aliases
        if ml in {"acc", "accuracy"}:
            nm = "accuracy"
        elif ml in {"balanced_acc", "balanced_accuracy"}:
            nm = "balanced_accuracy"
        elif ml in {"auc", "roc_auc", "auroc"}:
            nm = "roc_auc"
        elif ml in {"pr_auc", "auprc", "average_precision"}:
            nm = "pr_auc"
        elif ml in {"prec", "precision"}:
            nm = "precision"
        elif ml in {"rec", "recall"}:
            nm = "recall"
        elif ml in {"f1", "f1_score"}:
            nm = "f1"
        elif ml in {"f1_weighted", "f1_macro", "f1_micro"}:
            # pyhealth's binary only supports plain 'f1'; keep variants for multiclass
            nm = "f1" if is_binary else ml  # let multiclass pass through
        else:
            nm = m  # leave as-is; pyhealth will error if unsupported
        norm.append(nm)
        back.append(orig)
    return norm, back
def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    """
    Shape-robust + alias-robust wrapper around pyhealth metrics.
    - Uses pyhealth for everything (incl. balanced_accuracy).
    - Normalizes metric names to what pyhealth understands,
      then maps results back to the original names you passed in.
    """
    req = list(metrics) if isinstance(metrics, (list, tuple)) else [metrics]
    norm, back = _normalize_metric_names(req, is_binary=is_binary)

    if is_binary:
        y_prob, y_true = _standardize_binary_io(output, target)

        # AUROC guard (skip when only one class present)
        if ("roc_auc" not in norm) or (int(y_true.sum()) * (len(y_true) - int(y_true.sum())) != 0):
            raw = binary_metrics_fn(
                y_true,
                y_prob,
                metrics=norm,
                threshold=threshold,
            )
        else:
            raw = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Map back to requested keys
        out = {}
        for want, n in zip(back, norm):
            if n in raw:
                out[want] = float(raw[n])
        return out

    # multiclass
    y_out, y_true = _standardize_multiclass_io(output, target)
    raw = multiclass_metrics_fn(y_true, y_out, metrics=norm)

    out = {}
    for want, n in zip(back, norm):
        if n in raw:
            out[want] = float(raw[n])
    return out


@torch.no_grad()
def evaluate(data_loader, model, device, step_fn, header='Test:', metrics=['acc'], is_binary=True, use_amp=True):
    
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.to(device)
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0].to(device, non_blocking=True)
        target = batch[-1]
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = step_fn(model, EEG)
                loss = criterion(output, target)
        else:
            output = step_fn(model, EEG)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret