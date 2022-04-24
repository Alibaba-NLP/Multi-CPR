import logging
import os

from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RerankerDC if training_args.distance_cache else Reranker
    # print(config.num_labels)

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args
        )
        dev_dataset = PredictionDataset(
            data_args.dev_path, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )
    else:
        train_dataset = None
        dev_dataset = None


    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=GroupCollator(tokenizer),
    )
    if training_args.do_train:
        eval_qids = []
        eval_pids = []
        eval_labels = []
        with open(data_args.dev_id_file) as f:
            for line in f:
                line = line.strip().split()
                q = line[0]
                p = line[2]
                l = int(line[-1])
                eval_qids.append(q)
                eval_pids.append(p)
                eval_labels.append(l)
        trainer.eval_qids = eval_qids
        trainer.eval_pids = eval_pids
        trainer.eval_labels = eval_labels

    # Training
    if training_args.do_train:
        print('train_batch_size',training_args.train_batch_size)
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = PredictionDataset(
            data_args.pred_path, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        pred_labels = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                l = l.split()
                q, p  = l[0],l[2]
                pred_qids.append(q)
                pred_pids.append(p)
                # pred_labels.append(int(l[-1]))
                # print(trainer.evaluate(eval_dataset=test_dataset))

        eval_dataloader = trainer.get_eval_dataloader(test_dataset)

        output = trainer.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=False, 
            # metric_key_prefix=metric_key_prefix,
        )
        pred_scores = output.predictions
        # pred_scores = trainer.predict(test_dataset=test_dataset).predictions

	
        if trainer.is_world_process_zero():
            # assert len(pred_qids) == len(pred_scores)
            with open(data_args.rank_score_path, "w") as writer:
                for i,(qid, pid, score) in enumerate(zip(pred_qids, pred_pids, pred_scores)):
                    writer.write(f'{qid} Q0 {pid} {i} {score} {data_args.run_id}\n')
                    # writer.write(f'{qid}\t{pid}\t{score}\n')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
