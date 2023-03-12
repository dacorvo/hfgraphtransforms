# Evaluating the effect of graph transformations on a BeRT model

````
Usage:

usage: bert-evaluation.py [-h] [--device DEVICE] [--infer] [--evaluate]
                          [--transform]

HuggingFace Bert QuestionAnswering tester

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  Target device.
  --infer          Pass a simple question to the model.
  --evaluate       Evaluate the model score on the squad2 validation dataset.
  --transform      Test several model transformations.
````

Three transformations from optimum.fx.optimizations.transformation that are relevant
for a Transformer model are selected:
- MergeLinears,
- FuseBiasInLinear,
- ChangeTrueDivToMulByInverse.

Note: I could not apply the FuseBiasInLinear transform on the model due to a shape mismatch.
I think it happens in the score x value matmul, which is odd because my understanding is that Linear output
shapes should not be impacted (only inputs and weights).
I quickly checked the corresponding unit test in optimum and it is supposed to work out-of-the-box for Bert.

## Evaluation on a simple question

### Apple M1 Max - CPU

````
Base average inference: 597.0487041999999 ms
Base+LintAndRecompile average inference: 593.5082249999999 ms
Base+MergeLinears average inference: 596.3989666 ms
Base+ChangeTrueDivToMulByInverse average inference: 593.9272542000001 ms
Base+RemoveDropout average inference: 591.5362708 ms
````

### Apple M1 Max - Metal Performance Shaders (MPS)

````
Base average inference: 93.70445829999996 ms
Base+LintAndRecompile average inference: 93.08724579999996 ms
Base+MergeLinears average inference: 92.29938749999995 ms
Base+ChangeTrueDivToMulByInverse average inference: 93.05135 ms
Base+RemoveDropout average inference: 91.86703329999997 ms
````

## Evaluation on a subset of a squad dataset

### Apple M1 Max - CPU - 10 samples

````
Base : f1 = 30.00, samples/s = 0.3931, latency = 2543.8110 ms
````

### Apple M1 Max - Metal Performance Shaders (MPS) - 100 samples

````
Base : f1 = 43.89, samples/s = 9.6366, latency = 103.7709 ms
````

