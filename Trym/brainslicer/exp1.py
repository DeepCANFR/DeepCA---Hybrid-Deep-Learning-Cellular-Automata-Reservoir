from brainslicer import Brainslicer
import cupy as cp


class izk_experiment_1(Brainslicer):
    def __init__(self, array_provider=cp):
        self.array_provider=array_provider

    def run():
        soma = self.IzhikevichSoma()

if __name__ == "__main__":
    bs = izk_experiment_1()
    bs.run()