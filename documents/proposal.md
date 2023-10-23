# foreNNsic credit application proposal

We are a group of researchers investigating methods to infer properties of the execution environment of machine learning pipelines by tracing characteristic numerical deviations in observable outputs. We have shown that these deviations can be used forensically to identify the used CPU microarchitecture (published at ICASSP ‘21), and that we can amplify these differences to make them observable in a label-only setting (about to be presented at IH&MMSEC ‘21).
Our next goal is to extend these experiments to GPUs, as well as more CPU architectures and types. Our hope is to fully explain the root cause of these deviations. Another potential outcome is the generation of a large catalog of possible deviation characteristics, which in turn would allow forensic analysts to still gather information in scenarios where they don’t have full control over input, model, and output.

The use of cloud infrastructure allows us to quickly, reproducibly, and automatically create a number of machines with various hardware configurations. Not only does cloud infrastructure make this type of experiments feasible in the first place, it also drastically reduces the effort for other researchers to reproduce our results.

## GCP cost estimate

Can be found [here](https://cloud.google.com/products/calculator/#id=148ebf2f-75ce-46ee-8ea9-606517f8bdd6).
I put in one of each CPU and GPU type, with 3 hours on one day a week.
This leaves us at 137.62 per month. Almost half of that comes from one ultramem instance with 960GB memory.
That instance type is required for the Broadwell E7 CPU.
