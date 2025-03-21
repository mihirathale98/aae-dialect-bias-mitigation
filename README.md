# LoRA Fine-tuning for LLMs on Modal

This project allows you to fine-tune large language models (LLMs) such as LLaMA 2 and Phi using LoRA on Modal with A100 GPUs.

## Prerequisites

1. Install Modal CLI:
   ```bash
   pip install modal
   ```

2. Set up your Modal account and authenticate:
   ```bash
   modal token new
   ```

3. Prepare your dataset:
   - Create a CSV file with a column named `text` containing your training examples.
   - For instruction fine-tuning, format your data like this:
     ```csv
     text
     "<s>[INST] Your instruction here [/INST] Expected response here </s>"
     ```

## Usage

To run the fine-tuning process, use the following command:

```bash
python train_modal.py \
  --model_name "meta-llama/Llama-2-7b-chat-hf" \
  --dataset_path "path/to/your/dataset.csv" \
  --output_dir "./finetuned-model" \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_seq_length 2048
```

### Arguments

- `--model_name`: Name of the model to fine-tune (e.g., `meta-llama/Llama-2-7b-chat-hf` or `microsoft/phi-2`)
- `--dataset_path`: Path to your dataset CSV file
- `--output_dir`: Directory to save the fine-tuned model
- `--epochs`: Number of training epochs (default: `3`)
- `--batch_size`: Training batch size (default: `4`)
- `--learning_rate`: Learning rate for training (default: `2e-4`)
- `--lora_r`: LoRA rank (default: `16`)
- `--lora_alpha`: LoRA alpha (default: `32`)
- `--lora_dropout`: LoRA dropout (default: `0.05`)
- `--max_seq_length`: Maximum sequence length (default: `2048`)

## Notes

- Adjust the batch size and `max_seq_length` based on your GPU memory constraints.
- For LLaMA 2 7B model, an A100 with 40GB memory can handle batch sizes of 4-7 depending on sequence length.
- For Phi-2 (2.7B parameters), you can use larger batch sizes due to its smaller size.
- The script uses 4-bit quantization to reduce memory usage.
- Make sure you have the necessary permissions to access the specified models (especially for LLaMA 2).

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure you have the latest version of Modal CLI installed.
2. Verify that your Modal account is properly set up and authenticated.
3. Check that your dataset CSV file is formatted correctly.
4. If you're using LLaMA 2, make sure you have the required permissions from Meta.

For any other issues, please refer to the [Modal documentation](https://modal.com/docs) or open an issue in this repository.