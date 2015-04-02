
import theano

class DebugAbstraction:
    def __init__(self, **kwargs):
        self._debugprints = []

    def debugprint(self, message, content):
        self._debugprints.append(
            theano.printing.Print(message)(content)
        )

    def debugprint_list(self):
        return self._debugprints
