import argparse
import timeit
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import evaluator
from datasets import load_dataset


def infer_qa_model(model, tokenizer, inputs):
    """ Pass question answering inputs to a model

    Args:
        model (`PreTrainedModel`): the model to transform.
        tokenizer (`Tokenizer`): the model tokenizer.
        inputs (``): the context and question.

    Returns:
        `str`: The answer 
    """
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))


def evaluate_squad_qa_model(model, tokenizer, data, device):
    """ Evaluate the model on squad data for question answering.

    Args:
        model (`PreTrainedModel`): the model to evaluate.
        tokenizer (`Tokenizer`): the model tokenizer.
        data (obj:`Dataset`): the squad_v2 dataset.
        device (`str`): a string describing the target device.

    Returns:
        a dict of results
    """
    task_evaluator = evaluator("question-answering")
    return task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=data,
        metric="squad_v2",
        squad_v2_format=True,
        device=torch.device(device)
    )


def setup_squad_data(n_samples):
    return load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(n_samples))


def setup_qa_inputs(tokenizer, question, text, device):
    return tokenizer(question, text, return_tensors="pt").to(device)


def setup_qa_model(model_path, device):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='HuggingFace Bert QuestionAnswering tester')
    parser.add_argument('--device', help='Target device.', type=str, default='mps')
    parser.add_argument('--infer', help='Pass a simple question to the model.', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate the model score on the squad2 validation dataset.', action='store_true')
    args = parser.parse_args()
    model_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model, tokenizer = setup_qa_model(model_path, args.device)
    question, text = "Which zelda game is my favorite?", "I cannot decide whether my favorite Zelda game is Ocarina of time or The Wind Waker."
    inputs = setup_qa_inputs(tokenizer, question, text, args.device)
    if args.infer:
        print(text)
        print(question)
        answer = infer_qa_model(model, tokenizer, inputs)
        print(answer)
    if args.evaluate:
        n_samples = 100 if args.device == 'mps' else 10
        data = setup_squad_data(n_samples)
        results = evaluate_squad_qa_model(model, tokenizer, data, args.device)
        print(results)


if __name__ == "__main__":
    main()