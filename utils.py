import ast
import collections
import copy
import gc
import joblib
import math
import numpy as np
import os
import pandas as pd
import random
import torch

from collections import Iterable
from datasets import Dataset
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    nested_concat
)
from transformers.trainer_utils import (
    EvalPrediction, 
    PredictionOutput
)
from transformers import (
    BertForTokenClassification,
    PreTrainedTokenizer, 
    Trainer
)
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Optional
)

from toxic_spans.evaluation.semeval2021 import f1


class Converter:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 200, add_special_tokens: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Tokenize and converet Semeval data to NER-like data with two classes."""
        text_batch = examples['text']
        
        # Obtain tokens and token span intervals
        tokenized_inputs_batch = self.tokenizer(
            text_batch,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens=self.add_special_tokens,
            return_special_tokens_mask=self.add_special_tokens
        )
        token_spans_intervals_batch = tokenized_inputs_batch['offset_mapping']
        
        # Save token span intervals for later f1 calculation
        tokenized_inputs_batch['token_spans_intervals'] = token_spans_intervals_batch

        if 'spans' in examples:
            spans_batch = examples['spans']

            # Convert span literal to list
            spans_batch = [ast.literal_eval(spans) for spans in spans_batch]
        
            # Convert label spans to span intervals convert 
            # (label spans intervals, token span intervals) -> token labels
            tokenized_inputs_batch["labels"] = [
                Converter._intervals_to_token_labels(
                    token_spans_intervals, 
                    Converter._spans_to_span_intervals(spans, len(text))
                )
                for spans, text, token_spans_intervals in zip(spans_batch, text_batch, token_spans_intervals_batch)
            ]
            tokenized_inputs_batch['spans'] = spans_batch
        
        return tokenized_inputs_batch

    def _spans_to_span_intervals(spans: List[int], text_len: int) -> List[Tuple[int, int]]:
        """Convert span to span intervals"""
        if not spans:
            return []
        
        assert text_len != 0, f'Spans provided with text length == 0, spans = {spans}'
        assert max(spans) <= text_len, f'Max index in span out of range, ' \
            f'max(spans) > text_len : {max(spans)} > {text_len}'

        span_intervals = [(spans[0], None)]
        span_end_candidate = spans[0]
        for index in spans[1:]:
            if index == span_end_candidate + 1:
                span_end_candidate = index
                continue
            
            span_intervals[-1] = (span_intervals[-1][0], span_end_candidate + 1)
            span_end_candidate = index

            span_intervals.append((index, None))

        if spans[-1] != text_len - 1:
            span_intervals[-1] = (span_intervals[-1][0], span_end_candidate + 1)
        else:
            span_intervals[-1] = (span_intervals[-1][0], text_len)
        
        return span_intervals

    def _includes(a: Tuple[int, int], b: Tuple[int, int]):
        """True if a interval includes b interval False otherwise"""
        return a[0] <= b[0] and a[1] >= b[1]

    def _intervals_to_token_labels(
        token_spans_intervals: List[Tuple[int, int]], 
        label_spans_intervals: List[Tuple[int, int]]
    ) -> List[int]:
        """Convert token span intervals and label span intervals to list of labels"""
        token_labels = []
        for token_span_interval in token_spans_intervals:
            is_positive = int(any(
                Converter._includes(label_spans_interval, token_span_interval) 
                for label_spans_interval in label_spans_intervals
            ))
            token_labels.append(is_positive)
        return token_labels

    def _span_intervals_to_spans(span_intervals: List[Tuple[int, int]]) -> List[int]:
        """Convert span intervals to spans"""
        if not span_intervals:
            return []

        spans = []
        
        # Fill intervals
        for span_interval in span_intervals:
            span = list(range(*span_interval))
            spans.extend(span)

        # Merge spans
        unique_indices = set()
        merged_spans = []
        for index in spans:
            if index not in unique_indices:
                merged_spans.append(index)
                unique_indices.add(index)

        return merged_spans


class CustomSpansTrainer(Trainer):
    """Overloads Trainer's prediction_loop method to gather span info and provide it to compute_metrics"""
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: Optional[str] = 'eval'
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
            
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        spans_info = {
            'spans': [],
            'token_spans_intervals': [],
            'tokens_mask': [],
        }
        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

            if 'spans' in inputs:
                spans_info['spans'].extend(inputs['spans'])
            
            spans_info['token_spans_intervals'].extend(inputs['token_spans_intervals'])
            if 'special_tokens_mask' in inputs:
                spans_info['tokens_mask'].extend(inputs['attention_mask'] & (~inputs['special_tokens_mask']))
            else:
                spans_info['tokens_mask'].extend(inputs['attention_mask'])

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
    
        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids), spans_info)
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


class BertForTokenMultiLabelClassification(BertForTokenClassification):
    """Overloaded forward to use BCEWithLogitsLoss instead of CrossEnrtopyLoss"""
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.type_as(logits)
            flattened_logits = torch.flatten(logits)
            flattened_labels = torch.flatten(labels)
            # Only keep active parts of the loss
            if attention_mask is not None:
                flattened_attention_mask = torch.flatten(attention_mask).type_as(logits)
                loss_fct = BCEWithLogitsLoss(weight=flattened_attention_mask)
                loss = loss_fct(flattened_logits, flattened_labels)
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(flattened_logits, flattened_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compress_rec(data, selectors):
    """Filter N-dimentional iterable by mask.
    https://stackoverflow.com/questions/47793276/how-to-mask-2d-list-in-python
    """
    for d, s in zip(data, selectors):
        if isinstance(d, Iterable) and isinstance(s, Iterable):
            yield compress_rec(d, s)
        else:
            if s:
                yield d


def predict_token_spans(p, spans_info, threshold=0.5):
    """Compute f1 score from classifier predictions, spans and token span intervals"""
    tokens_mask = torch.stack(spans_info['tokens_mask']).cpu().numpy().astype(bool)

    # Get predictions from logits
    predictions, labels = p
    predictions = np.squeeze((expit(predictions) >= threshold).astype(np.int8))

    # Filter out masked and special tokens 
    predictions[~tokens_mask] = 0
    token_spans_intervals_filtered = compress_rec(spans_info['token_spans_intervals'], predictions)

    # Convert
    token_spans_predicted = map(Converter._span_intervals_to_spans, token_spans_intervals_filtered)

    return token_spans_predicted


def compute_metrics(p, spans_info, threshold=0.5, raw=False):
    """Compute f1 score from classifier predictions, spans and token span intervals"""
    # Convert
    token_spans_predicted = predict_token_spans(p, spans_info, threshold)

    # Compute f1 for each sample and average it
    f1_char = [f1(token_spans_p, token_spans_l) for token_spans_p, token_spans_l in zip(token_spans_predicted, spans_info['spans'])]
    f1_char_mean = np.mean(f1_char)

    metrics = {
        'f1_char_mean': f1_char_mean,
    }

    if raw:
        metrics['f1_char'] = f1_char

    return metrics


def extract_predictions_as_metric(p, spans_info, threshold=0.5):
    token_spans_predicted = predict_token_spans(p, spans_info, threshold)
    return {'token_spans_predicted': list(token_spans_predicted)}


def build_bert_params_decayed_lr(model, lr, decay_factor=1):
    """
    Build decayed LR for each token-classification BERT layer
    see https://arxiv.org/abs/1905.05583
    """
    params = [
        {'params': model.bert.embeddings.parameters(), 'lr': lr * decay_factor ** 13},
        *[{'params': model.bert.encoder.layer[i].parameters(), 'lr': lr * decay_factor ** (12 - i)} for i in range(11)],
        {'params': model.classifier.parameters(), 'lr': lr},
    ]

    return params


def get_cosine_schedule_with_init_const_after_const(
    optimizer, num_init_const_steps, num_training_steps, num_const_after_steps, num_cycles=0.5, last_epoch=-1
):
    """Build LR scheduler"""
    def lr_lambda(current_step):
        if current_step < num_init_const_steps:
            return 1.0
        elif current_step < num_const_after_steps:
            progress = float(current_step - num_init_const_steps) / float(max(1, num_training_steps - num_init_const_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        else:
            progress = float(num_const_after_steps - num_init_const_steps) / float(max(1, num_training_steps - num_init_const_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def keep_fields_data_collator(features: List[Any], keep_keys: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Overload transformers.data.data_collator.default_data_collator 
    to be able not to convert particular fields to tensors.
    """
    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    keep = ["label", "label_ids"]
    if keep_keys is not None:
        keep += keep_keys
    
    for k, v in first.items():
        if k not in keep:
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
        else:
            batch[k] = [f[k] for f in features]

    return batch


def reseed_all(seed_value):
    """
    Set seed for reproducibility of results.
    Use before each model initialization.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir, n_splits=5, seed=0):
    """Load train and trial data, merge it, drop duplicates and provide KFold indices
    either loaded from disk or generated with given seed.
    """
    # Load and concat train & trial
    df = pd.concat([pd.read_csv(f'{data_dir}/tsd_train.csv'), pd.read_csv(f'{data_dir}/tsd_trial.csv')], axis=0).reset_index()
    
    # Remove text duplicates to avoid data-leaks
    df = df.drop_duplicates(subset=['text'])

    # Get train / val split indices either from new KFold or from disk
    if not os.path.exists('split_indices.joblib'):
        indices = list(range(df.shape[0]))
        train, valid, indices_train, indices_valid = train_test_split(df, indices, test_size=0.14, random_state=seed)
        train, meta, indices_train, indices_meta = train_test_split(train, indices_train, test_size=0.17, random_state=seed)
        kfold = KFold(n_splits, shuffle=True, random_state=seed)
        split_indices = [indices_train, indices_valid, indices_meta] + list(kfold.split(train))
        joblib.dump(split_indices, 'split_indices.joblib')
    else:
        split_indices = joblib.load('split_indices.joblib')
        assert (len(split_indices[0]) + len(split_indices[1]) + len(split_indices[2])) == len(df)

    return df, split_indices


def cross_validate(build_trainer, dataset, split_indices, seed=0):
    """Perform cross-validation on a given k-fold split.
    build_trainer: function to build trainer. 
        hint: use closured function in main.py / jupyter notebook 
        to capture all the trainer args into build_trainer
    dataset: datasets.Dataset
    split_indices: list of folds
    """
    f1_scores_folds = []
    for i, (train_indices, test_indices) in enumerate(split_indices):
        # Build train / test datasets
        dataset_train, dataset_test = Dataset.from_dict(dataset[train_indices]), Dataset.from_dict(dataset[test_indices])

        # Split on train and val
        dataset_train = dataset_train.train_test_split(0.2, shuffle=True, seed=seed)

        # Build trainer
        trainer = build_trainer(f'fold_{i}', dataset_train['train'], dataset_train['test'])

        # Train
        trainer.train()

        # Eval on validation
        f1_score = trainer.evaluate(dataset_test)['eval_f1_char_mean']
        f1_scores_folds.append(f1_score)

        print(f'Fold {i + 1} / {len(split_indices)}, mean F1 score: {f1_score}')
    
        # Clean memory
        del trainer, dataset_train, dataset_test
        torch.cuda.empty_cache()
        gc.collect()

    return f1_scores_folds
    