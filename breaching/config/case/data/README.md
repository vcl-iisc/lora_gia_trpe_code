### Commom parameters

| Variable | Type    | Description                                                                                               |
|----------|---------|-----------------------------------------------------------------------------------------------------------|
| name     | string  | Defaults to the name of the dataset itself.                                                               |
|          |         |                                                                                                           |
| modality | enum    | "vision" or "text", based in the dataset nature.                                                          |
|          |         |                                                                                                           |
| task     | enum    | Vision datasets: defaults to "classification".                                                            |
|          |         | Text datasets: can be "masked-lm" or  "causal-lm".                                                        |
|          |         |                                                                                                           |
| path     | string  | The path of the dataset in the system. Defaults to "~/data".                                              |
|          |         |                                                                                                           |
| size     | int     | The size of the data in the dataset                                                                       |
|          |         |                                                                                                           |
| shape    | 3-tuple | Vision datasets: it is a 3-tuple of (channel, width, heigth) of the images on the dataset                 |
|          | int     | Text datasets: it is the sequence_lenght of the dataset, that represents the number of tokens/words in it |

#### Federated Learning specifics:

| Variable            | Type | Description                                                       |
|---------------------|------|-------------------------------------------------------------------|
| default_clients     | int  | Estimate on number of articles in dataset                         |
|                     |      |                                                                   |
| partition           | enum | Used by server module when preparing the dataset for use.         |
|                     |      | given: use natural data partition. Default text dataset partition |
|                     |      | balanced:                                                         |
|                     |      | unique-class:                                                     |
|                     |      | mixup:                                                            |
|                     |      | feat_est:                                                         |
|                     |      | random-full:                                                      |
|                     |      | random:                                                           |
|                     |      | none:                                                             |
|                     |      |                                                                   |
| examples_from_split | enum | Used by server module when preparing the dataset for use.         |
|                     |      | train:                                                            |
|                     |      | training:                                                         |
|                     |      | validation:                                                       |

#### Data-specific implementation constants:

| Variable   | Type    | Description                                                                                           |
|------------|---------|-------------------------------------------------------------------------------------------------------|
| batch_size | int     | number of samples processed together in one pass                                                      |
|            |         |                                                                                                       |
| caching    | boolean |                                                                                                       |
|            |         |                                                                                                       |
| db         | enum    | Can be "none" or "LMDB".                                                                              |
|            |         | Database Setup. Use the cmd-line hydra overload feature to activate the LMDB module with data.db=LMDB |

<hr />

### Vision datasets specific parameters:

| Variable | Type | Description                                 |
|----------|------|---------------------------------------------|
| classes  | int  | The number of labels present in the dataset |

#### Preprocessing:

| Variable  | Type    | Description                                           |
|-----------|---------|-------------------------------------------------------|
| normalize | boolean | Scales pixel values to an standard range              |
|           |         |                                                       |
| mean      | 3-tuple | Average pixel values across the entire dataset        |
|           |         |                                                       |
| std       | 3-tuple | Standard deviation of pixel values across the dataset |

#### Data Augmentations:

| Variable                                 | Type   | Description                                                       |
|------------------------------------------|--------|-------------------------------------------------------------------|
| augmentations_train                      | object |                                                                   |
| augmentations_train.RandomCrop           | tuple  | Randomly crops an image to the specified size to create variation |
| augmentations_train.RandomHorizontalFlip | float  | Ramdomly flips image left-right                                   |
|                                          |        |                                                                   |
| augmentations_val                        | object |                                                                   |
| augmentations_val.Resize                 | int    | Resizes image to fixed dimensions                                 |
| augmentations_val.CenterCrop             | int    | Crops center portion to exact size for consistent evaluation      |

<hr />

### Text datasets specific parameters:

#### Only used when task=masked-lm:

| Variable        | Type    | Description                                                                                                      |
|-----------------|---------|------------------------------------------------------------------------------------------------------------------|
| mlm_probability | float   | Defaults to "0.15". It is the probability that any given token will be masked in Masked Language Modeling (MLM). |
|                 |         |                                                                                                                  |
| disable_mlm     | boolean | Defaults to "False". Disables the MLM.                                                                           |

#### Preprocessing:

| Variable   | Type | Description                                   |
|------------|------|-----------------------------------------------|
| tokenizer  | enum | Model used for split text into smaller units. |
|            |      | Can be "GPT-2" and "bert-base-uncased"        |
|            |      |                                               |
| vocab_size | int  | Number of unique tokens in model dictionary.  |

