class bertscore_config:
    metric_name = "bertscore"
    metric_path = "bertscore"

    uses_reference = True
    uses_source = False

    score_name = "f1"

    load_args = {}
    compute_args = {
        "model_type": "bert-base-uncased",
        "lang": "en",
    }

class bleurt_config:
    metric_name = "bleurt"
    metric_path = "bleurt"

    uses_reference = True
    uses_source = False

    score_name = "scores"

    load_args = {
        "config_name": "bleurt-base-512"
    }
    compute_args = {}

class comet_config:
    metric_name = "comet"
    metric_path = "comet"

    uses_reference = True
    uses_source = True

    score_name = "scores"

    load_args = {}
    compute_args = {"progress_bar": True}

class frugalscore_config:
    metric_name = "frugalscore"
    metric_path = "frugalscore"

    uses_reference = True
    uses_source = False

    score_name = "scores"

    load_args = {"config_name": "moussaKam/frugalscore_tiny_bert-base_bert-score"}
    compute_args = {
        "max_length": 512,
        "batch_size": 128,
        "device": "gpu"
    }

class bartscore_config:
    metric_name = "bartscore"
    metric_path = "./geneval/geneval/metrics/bartscore/bartscore.py"

    uses_reference = False
    uses_source = True

    score_name = None

    load_args = {}
    compute_args = {
        #"model_type": "facebook/bart-base",
        "model_type": "facebook/bart-large-cnn",
        "max_length": 512,
        "batch_size": 128,
    }

class moverscore_config:
    metric_name = "moverscore"
    metric_path = "./geneval/geneval/metrics/moverscore/moverscore.py"

    uses_reference = True
    uses_source = False

    score_name = None

    load_args = {}
    compute_args = {
        "model_type": "bert-base-uncased",
        "use_idfs": True,
        "batch_size": 128,
        "n_gram": 1,
    }

class baryscore_config:
    metric_name = "baryscore"
    metric_path = "./geneval/geneval/metrics/baryscore/baryscore.py"

    uses_reference = True
    uses_source = False

    score_name = "baryscore_W"

    load_args = {}
    compute_args = {
        "model_type": "bert-base-uncased",
        "batch_size": 256,
        "last_layers": 5,
        "use_idfs": True,
        "sinkhorn_ref": 0.01
    }
