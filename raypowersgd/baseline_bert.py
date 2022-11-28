from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from torch.utils.data import DataLoader

def tokenize_func(data, tokenizer):
    return tokenizer(data["text"], padding="max_length", truncation=True)


def main():

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

    train_dataloader = DataLoader(small_train_set, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_set, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # =====TRAIN=====

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # =====TEST=====

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == "__main__":
    main()