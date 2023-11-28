import glob
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def train_model(data_path, model_path, output_dir, num_train_epochs=3):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Detai\HelixEsq")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Preparing the dataset
    files = glob.glob(os.path.join(data_path, '**/*.txt'), recursive=True)
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=files,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,  # Adjust according to your GPU memory
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Training
    trainer.train()

    # Saving the model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    data_path = r'C:\Users\Detai\Documents'  # Path to your dataset
    model_path = r'C:\Users\Detai\HelixEsq'  # Path to your model
    output_dir = r'C:\Users\Detai\HelixEsqTrained'  # Where to save the trained model

    train_model(data_path, model_path, output_dir)

    