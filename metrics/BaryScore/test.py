import datasets

if __name__ == '__main__':
    """
    Here you can find an example to use the BaryScore
    """
    ref = ['I like my cakes very much',
           'I hate these cakes!']
    hypothesis = ['I like my cakes very much',
                  'I like my cakes very much']

    s = datasets.load_metric("baryscore.py")
    s.add_batch(predictions=hypothesis, references=ref)
    t = s.compute()
    print(t)
