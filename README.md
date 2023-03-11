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
Base average inference: 595.8505375000001 ms
Base+MergeLinears average inference: 595.4529500000001 ms
Base+ChangeTrueDivToMulByInverse average inference: 592.7251125000001 ms
````

### Apple M1 Max - Metal Performance Shaders (MPS)

````
Base average inference: 89.81496250000002 ms
Base+MergeLinears average inference: 86.32195840000004 ms
Base+ChangeTrueDivToMulByInverse average inference: 89.87490410000002 ms
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

