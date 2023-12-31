# Bengali Person Name Extractor
This project aims to build a person-name extractor for Bengali text. The model takes a sentence as input and outputs the person name present in the input sentence, or handles cases where no person's name is present.

Example -

``আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম``
From the above sentence, the model should extract ``আব্দুর রহিম``

## Important versions
We use the following versions of softwares for the V2 system.

|software|version|
|-|-|
|python|3.8.6|
|pip|23.3.1|

## Dataset
The dataset comprises 20 text files, each containing sentences alongside corresponding tag/entity. In each file, words are paired with their respective tags. The dataset adheres to the BIO format, where an entity starts with B (Beginning). For multi-word entities, the second and subsequent words carry an I (Inside) tag. The O (Outside) tag serves as a default label for words lacking specific annotations. Here's an illustrative example:

```
Word    Tag
---------
অচল	O
হয়ে	O
পড়ছে	O
দেশের	B-LOC
অর্থনীতি	O
,	O
কমছে	O
প্রবৃদ্ধি	O
।	O

```
In the provided example, "দেশের" constitutes a complete location. The initial word is labeled as B-LOC, and subsequent words, such as "অর্থনীতি," are marked as O since they do not correspond to any specific tag.

## Thought Process
My first thought was it is Named-entity recognition (NER). And Machine Learning wise it is classification problem. I treated it as a classification problem. To gain a better understanding of the data, I loaded the dataset and performed visualization, revealing an imbalance issue in the distribution plot.

Also our focus is person name so I removed the samples without person name and did under-sampling for tag like `O`. That help to solve imbalance issue. I skipped standard pre-processing like stemming because avaible libs not working. One common issue with them is - `অমর্ত্য সেন` became -> `অমর্ত্য স` after stemming.

Person name average length is about 2. So I took only 3 surrunding words/token from both side of candidate word as feature. As you know we can not use `str` as features we need math representation for word/token. Word2vec is smart way to represent word. I used word2vec.

For the model, I chose a simple multi-layered Artificial Neural Network (ANN). 2 reasons behind this decision. First one is faster training and better performance. secound one is small dataset.



## How to train
To train the model from scratch, follow these steps:
1. Install the required Python packages by running the following command:
    ```bash
    pip install -r requirements.txt
    ```
2. Execute the training script (training.py) to initiate the model training process:
    ```bash
    python training.py
    ```

## How to get name from sentence
1. If requirements are not installed. Install the required Python packages by running the following command:
    ```bash
    pip install -r requirements.txt
    ```
2. Get names from sentence. Sentence need to pass inside qutation.
    - Example with name not exist is both training and testing dataset:
    ```bash
    python predict.py "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম"
    ```

    Output:
    ```
    Given sentence: আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম
    Extracted names: আব্দুর রহিম
    ```

    - Example with multiple names:
    ```bash
    python predict.py "যুবলীগের কর্মী মাসুম ছাত্রদলের কর্মী সাদ্দামকে গুলি করেন"
    ```

    Output:
    ```
    Given sentence: যুবলীগের কর্মী মাসুম ছাত্রদলের কর্মী সাদ্দামকে গুলি করেন
    Extracted names: মাসুম ,সাদ্দামকে
    ```
    - Example without name:
    ```bash
    python predict.py "বন্দর থানার ভারপ্রাপ্ত কর্মকর্তা"
    ```

    Output:
    ```
    Given sentence: বন্দর থানার ভারপ্রাপ্ত কর্মকর্তা
    Extracted names: No name
    ```

3. If you face issue with running code in terminal. Then try this notebook (`predict.ipynb`).


## Contact
For any questions or further clarification, feel free to contact:


Email: NurulAkterTowhid@gmail.com
Thank you for considering this submission!