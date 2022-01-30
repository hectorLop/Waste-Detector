class Config(dict):
    """
    Config class that defines training parameters and hyperparameters.
    """

    def __init__(self, **kwargs):
        super(self, dict).__init__(kwargs)