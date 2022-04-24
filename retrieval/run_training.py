import os
import json
import math
import sys
import logging
from datasets import load_dataset

from data import TextMatchingCollator, TextMatchingDataset
from modeling import TextMatchingForBert
import transformers
from transformers import Trainer
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process
from arguments import DataTrainingArguments, ModelArguments, \
    TextMatchingForBertTrainingArguments as TrainingArguments


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

TextMatch_TYPE_MAP = {
    'bert': TextMatchingForBert
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

     # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


    print("model_args: ",model_args)
    print("training_args: ", training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)
    train_set = TextMatchingDataset(load_dataset(
        'json',
        #path = "/home/dingkun.ldk/dev/vector_model/utils/json.py",
        data_files=data_args.train_path,
        block_size=2 ** 25,
        ignore_verifications=False,
    )['train'], data_args)
    dev_set = None
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir, use_fast=False
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # initialize the Condenser Pre-training LMX
    if model_args.model_type not in TextMatch_TYPE_MAP:
        raise NotImplementedError(f'Condenser for {model_args.model_type} LM is not implemented')
    _condenser_cls = TextMatch_TYPE_MAP[model_args.model_type]
    if model_args.model_name_or_path:
        model = _condenser_cls.from_pretrained(model_args.model_name_or_path, pool_type=model_args.pool_type, model_args=model_args)
    else:
        logger.warning('Training from scratch.')
        model = _condenser_cls.from_config(
            config, model_args, data_args, training_args)

    model.bert.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = TextMatchingCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
