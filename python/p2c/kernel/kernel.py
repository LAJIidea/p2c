class Kernel:
    counter = 0

    def __init__(self, _func, _classkernel=False) -> None:
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        self.arguments = []
        self.return_type = None
        self.classkernel = _classkernel
        
