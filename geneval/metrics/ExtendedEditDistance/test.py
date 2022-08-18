import datasets

if __name__ == '__main__':
    """
    Here you can find an example to use the ExtendedEditDistance
    """
    hypothesis = "Today the weather is very nice."
    reference = "Today the weather is nott nice."
    metric = datasets.load_metric("eed.py")
    score = metric.compute(predictions=hypothesis, references=reference)
    print(score)
