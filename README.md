# Sentiment-Classification-on-the-IMDb-Dataset  BERT Fine-Tuning for Sentiment Classification on IMDB Dataset

## Project Overview
This project involves fine-tuning a pre-trained BERT model (`bert-base-uncased`) for binary sentiment classification using the IMDB Movie Reviews dataset. The dataset contains movie reviews labeled as either 'positive' or 'negative'. The goal is to fine-tune a transformer model to classify reviews based on their sentiment.

## Steps Involved

### 1. **Install Required Libraries**
We begin by installing the necessary libraries for the project. This includes the Hugging Face `transformers` and `datasets` libraries for pretrained models and datasets.

```bash
pip install transformers datasets
pip install --upgrade pip transformers datasets huggingface_hub fsspec
```

### 2. **Import Python Libraries**
We import essential Python libraries for the task including PyTorch, Hugging Face utilities, sklearn metrics, and more.

```python
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
```

### 3. **Load the IMDB Dataset**
The IMDB dataset is loaded from the Hugging Face datasets library. It is a binary sentiment classification task with 'positive' and 'negative' reviews.

```python
dataset = load_dataset("imdb")
```

### 4. **Preprocess the Dataset with BERT Tokenizer**
The text data is preprocessed using the BERT tokenizer (`bert-base-uncased`), which is responsible for converting text into tokens suitable for input into the model.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

### 5. **Define Evaluation Metrics**
The evaluation metrics include accuracy, precision, recall, and F1-score. These metrics are used to evaluate the model's performance during the training and evaluation phases.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
```

### 6. **Model Fine-Tuning**
The pre-trained BERT model is fine-tuned on the IMDB dataset with the following hyperparameters:
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 2

```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results_A",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs_A"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_datasets["test"].select(range(1000)),
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 7. **Evaluate the Model**
After fine-tuning the model, we evaluate its performance on the test set. The evaluation metrics (accuracy, precision, recall, F1-score) are calculated.

```python
metrics = trainer.evaluate()
print(metrics)
```

### 8. **Hyperparameter Tuning**
Three configurations are compared with different learning rates, batch sizes, and epochs to determine the best-performing configuration.

- **Config A**: Learning Rate = 2e-5, Batch Size = 16, Epochs = 2
- **Config B**: Learning Rate = 5e-5, Batch Size = 32, Epochs = 3
- **Config C**: Learning Rate = 3e-5, Batch Size = 16, Epochs = 4

```python
def train_model(learning_rate, batch_size, epochs, label):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir=f"./results_{label}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f"./logs_{label}",
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(5000)),
        eval_dataset=tokenized_datasets["test"].select(range(1000)),
        compute_metrics=compute_metrics,
    )

    print(f"
🧪 Training {label} with LR={learning_rate}, BS={batch_size}, Epochs={epochs}")
    trainer.train()
    metrics = trainer.evaluate()
    return label, metrics
```

### 9. **Results**
The results of all three configurations are displayed with accuracy, F1-score, and other metrics. You can compare the models' performance and choose the optimal configuration for your use case.

```python
results = []
results.append(train_model(2e-5, 16, 2, "A"))
results.append(train_model(5e-5, 32, 3, "B"))
results.append(train_model(3e-5, 16, 4, "C"))

# Show all metrics
for label, metric in results:
    print(f"
📊 {label} -> Accuracy: {metric['eval_accuracy']:.4f}, F1: {metric['eval_f1']:.4f}")
```

## Results and Conclusion
- The model is evaluated on various configurations to find the optimal performance.
- **Config B** with Learning Rate = 5e-5, Batch Size = 32, and Epochs = 3 yielded the best results.
- Future work can focus on deploying the model and further fine-tuning with additional datasets.

## Running the Project
To run this project on your local machine, ensure that you have the following Python packages installed:

```bash
pip install torch transformers datasets scikit-learn matplotlib
```

Then, simply execute the provided Python code for fine-tuning and evaluation.

## License
This project is licensed under the MIT License.
