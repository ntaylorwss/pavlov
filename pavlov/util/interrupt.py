import signal


class AtomicLoop:
    """A sigint handler that first finishes the iteration it's on, then terminates cleanly.

    This is useful for running an agent indefinitely, since it can be stopped
    between episodes cleanly with a KeyboardInterrupt.
    Credit: @kysely, https://github.com/kysely
    """
    def _handler(self, sig, frame):
        self.run = False
        print('SIGINT({}) recognized; finishing current episode before terminating'.format(sig))

    def __enter__(self):
        self.run = True
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *args):
        print("Terminating.")
