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
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Preparing the dataset
    files = glob.glob(os.path.join(data_path, '**/*.txt'), recursive=True)

    def _combine_files(file_list, output_file="combined_dataset.txt"):
        with open(output_file, "w", encoding="utf-8") as out_f:
            for f in file_list:
                with open(f, "r", encoding="utf-8") as in_f:
                    out_f.write(in_f.read())
                    out_f.write("\n")
        return output_file

    if not files:
        raise ValueError(f"No .txt files found in {data_path}")

    combined_path = _combine_files(files)
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=combined_path,
        block_size=128
    )
    os.remove(combined_path)

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

    