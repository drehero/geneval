import datasets

if __name__ == '__main__':
    """
    Here you can find an example to use the TransQuest
    """
    source = "Today the weather is very nice."
    target = "Heute ist das Wetter sehr gut."
    metric = datasets.load_metric("transquest.py")
    metric.add(source=source, target=target)
    score = metric.compute()
    print(score)
