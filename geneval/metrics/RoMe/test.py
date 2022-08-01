import datasets

if __name__ == '__main__':
    """
    Here you can find an example to use the RoMe
    """
    hyp = "103 hera discoverer james craig watson james craig watson deathcause peritonitis"
    ref = "james craig watson , who died of peritonitis , was the discoverer of 103 hera ."

    s = datasets.load_metric("rome.py")
    s.add(predictions=hyp, references=ref)
    t = s.compute()
    print(t)
