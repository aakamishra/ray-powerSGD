import argparse
import time

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from torch.utils.data import DataLoader
from powersgd import optimizer_step, PowerSGD, Config

import ray
import ray.train as train
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer


# Training BERT using PowerSGD

def tokenize_func(data, tokenizer):
    return tokenizer(data["text"], padding="max_length", truncation=True)


def ray_train_epoch(model, device, train_dataloader, optimizer, powersgd, lr_scheduler, epoch):
    start = time.time_ns()
    model.train()

    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        with model.no_sync():
            loss.backward()

        optimizer_step(optimizer, powersgd)
        lr_scheduler.step()
        # optimizer.zero_grad()

        if batch_idx % 100 == 0:
            print(f"Train Epoch={epoch}, Batch={batch_idx}/{len(train_dataloader)} \tLoss={loss.item()}")

    epoch_time = time.time_ns() - start
    print(f"Epoch time: {epoch_time}")
    return epoch_time


def ray_eval(model, device, eval_dataloader):
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()


def worker_train_func(config):
    # task = "sst2"
    # raw_data = load_dataset("glue", task)
    # metric = load_metric("glue", task)

    raw_data = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_data = raw_data.map(lambda x : tokenize_func(x, tokenizer) , batched=True)

    tokenized_data = tokenized_data.remove_columns(["text"])
    tokenized_data = tokenized_data.rename_column("label", "labels")
    tokenized_data.set_format("torch")

    small_train_set = tokenized_data["train"].shuffle(seed=42).select(range(1000))
    small_eval_set = tokenized_data["test"].shuffle(seed=42).select(range(1000))
    full_train_set = tokenized_data["train"]
    full_eval_set = tokenized_data["test"]

    batch_size = config["batch_size"]
    worker_batch_size = batch_size // train.world_size()

    train_dataloader = DataLoader(small_train_set, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_set, shuffle=False, batch_size=worker_batch_size)
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    eval_dataloader = train.torch.prepare_data_loader(eval_dataloader)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    model = train.torch.prepare_model(model)
    model.to(device)


    lr = config["lr"]
    params = model.parameters()
    optimizer = AdamW(params, lr=lr)
    powersgd = PowerSGD(list(params), config=Config(
        rank=1,  # lower rank => more aggressive compression
        min_compression_rate=10,  # don't compress gradients with less compression
        num_iters_per_step=2,  #   # lower number => more aggressive compression
        start_compressing_after_num_steps=0,
    ), device=device)

    num_epochs = config["epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )


    accuracy_results = []
    # os.environ["WANDB_API_KEY"] = "8f7086db96f9edfde9aae91cfcf98f1f445333f5"
    # wandb.init(project="powersgd-resnet-trial")

    # =====TRAIN AND TEST LOOP=====
    for epoch in range(num_epochs):
        epoch_time = ray_train_epoch(model, device, train_dataloader, optimizer, powersgd, lr_scheduler, epoch)
        accuracy = ray_eval(model, device, eval_dataloader)
        # checkpoint = TorchCheckpoint.from_state_dict(model.module.state_dict())
        # metrics = {"accuracy": accuracy, 'epoch': epoch, "time": epoch_time}
        # wandb.log(metrics)
        # session.report(metrics, checkpoint=checkpoint)
        accuracy_results.append(accuracy)
        print(f"Epoch={epoch}, accuracy={accuracy}")

    return accuracy_results


if __name__ == "__main__":

    # =====ARGUMENTS=====
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Enables GPU training")
    args, _ = parser.parse_known_args()
    print(">>> args: ", args)


    ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
    print(">>> Ray init done")

    scaling_config = ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu)
    trainer = TorchTrainer(
        train_loop_per_worker=worker_train_func,
        train_loop_config={
            "lr": 5e-5,
            "batch_size": 4,
            "epochs": 10
        },
        scaling_config=scaling_config
        # datasets={"train": train_dataset}
        )
    
    trainer.fit()