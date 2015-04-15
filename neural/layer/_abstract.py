
class LayerAbstract:
    def __init__(self, *args, **kwargs):
        self.weights = []
        self.outputs_info = []

    def infer_taps(self):
        taps = 0

        for info in self.outputs_info:
            # If info is dict it can have a taps array, default this is [-1]
            if (isinstance(info, dict)):
                # However of no `inital` property is provided it is treated
                # as None (no taps)
                if ('initial' in info):
                    if ('taps' in info):
                        taps += len(info.taps)
                    else:
                        taps += 1
            # If info is a numpy array or a scalar
            elif (info is not None):
                taps += 1

        return taps

    def setup(self, *args, **kwargs):
        raise NotImplemented

    def scanner(self, *args, **kwargs):
        raise NotImplemented