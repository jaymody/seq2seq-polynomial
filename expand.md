# Scale AI NLP Exercise - Polynomial Expansion

### Problem Statement

Implement a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as input and predict the expanded sequence. This is an exercise to demonstrate your machine learning prowess, so please refrain from parsing or rule-based methods.

A training file is provided in S3:
* `train.txt` : https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt

Each line of `train.txt` is an example, the model should take the factorized form as input, and predict the expanded form. E.g.

* `n*(n-11)=n**2-11*n`
* `n*(n-11)` is the factorized input
* `n**2-11*n`  is the expanded target

The expanded expressions are commutable, but only the form provided is considered as correct.

### Deliverables

Please submit a zip file with the following included:
* A `requirements.txt` including the packages and versions used in the solution
* A `network.txt` file that summarize the model architecture with the number of trainable parameters at each layer.
	* For TensorFlow, a simple `Model.summary()` will suffice.
	* For pyTorch, consider [`torchsummary`](https://github.com/sksq96/pytorch-summary).
* The training code
* A set of trained weights for the model
* A working `main.py` that evaluates the score of the trained model on `valid.txt`

### Evaluation

The model performance will be evaluated on a blind test set, using `main.py`. The submission will be evaluated on the following criteria:

   * Performances on a blind test set using the included scoring function. The minimum passing score is 0.7 but we would recommend pushing the model to its best possible performance.
   * Model design choices (e.g. architecture, network size, regularization)
   * Python proficiency and implementation details

Passing the minimum score threshold is not a guarantee of acceptance, all three factors will be considered for evaluation.

### Tips
* The maximum length of input or output sequences is 29.
* Do not use the unnecessarily large model; anything with more than 5M trainable parameters will be penalized.
