import argparse
import timeit
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.utils.fx import symbolic_trace
from optimum.fx.optimization.transformations import (MergeLinears,
                                                     FuseBiasInLinear,
                                                     ChangeTrueDivToMulByInverse,
                                                     LintAndRecompile,
                                                     compose)
from evaluate import evaluator
from datasets import load_dataset

from transformations import RemoveDropout


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
    start = torch.argmax(outputs['start_logits'])
    end = torch.argmax(outputs['end_logits'])
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))


def transform_model(model, transformation):
    """ Apply the specifed Transformation on a model.

    Args:
        model (`PreTrainedModel`): the model to transform.
        transformation (`Transformation`): the transformation to apply.

    Returns:
        `torch.fx.GraphModule`: The transformed model module
    """
    traced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
    )
    return transformation(traced)


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
    parser.add_argument('--transform', help='Test several model transformations.', action='store_true')
    args = parser.parse_args()
    model_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model, tokenizer = setup_qa_model(model_path, args.device)
    question, text = "Which zelda game is my favorite?", "I cannot decide whether my favorite Zelda game is Ocarina of time or The Wind Waker."
    inputs = setup_qa_inputs(tokenizer, question, text, args.device)
    transforms = [LintAndRecompile(), MergeLinears(), ChangeTrueDivToMulByInverse(), RemoveDropout()]
    all_transforms = compose(*transforms)
    transforms += [all_transforms]
    if args.infer:
        print(text)
        print(question)
        print(infer_qa_model(model, tokenizer, inputs))
        prefix = "Base"
        if args.transform:
            for transform in [None] + transforms:
                new_model = model if transform is None else transform_model(model, transform)
                t = timeit.timeit(lambda: infer_qa_model(new_model, tokenizer, inputs), number=10)/10
                prefix = "Base" if transform is None else f"Base+{transform.__class__.__name__}"
                print(f"{prefix} average inference: {t*1000} ms")
    if args.evaluate:
        n_samples = 100 if args.device == 'mps' else 10
        data = setup_squad_data(n_samples)
        results = evaluate_squad_qa_model(model, tokenizer, data, args.device)
        print(f"f1 = {results['f1']:.2f}")
        def summary(results):
            result = f"f1 = {results['f1']:.2f}"
            result += f", samples/s = {results['samples_per_second']:.4f}"
            result += f", latency = {results['latency_in_seconds'] * 1000:.4f} ms"
            return result
        if args.transform:
            # Unfortunately I could make it work because evaluate does not accept the GraphModule
            #for transform in [None] + transforms:
            for transform in [None]:
                new_model = model if transform is None else transform_model(model, transform).graph
                results = evaluate_squad_qa_model(new_model, tokenizer, data, args.device)
                prefix = "Base" if transform is None else f"Base+{transform.__class__.__name__}"
                print(f"{prefix} : {summary(results)}")


if __name__ == "__main__":
    main()