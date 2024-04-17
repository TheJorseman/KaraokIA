from typing import Callable, Any, List

class Compose(object):
    def __init__(
        self,
        transforms: List[Callable],
        ) -> None:
        self.transforms = transforms
    
    def __call__(self, _input, *args: Any, **kwds: Any) -> Any:
        for transform in self.transforms:
            _input = transform(_input, *args, **kwds)
        return _input

class Mono(object):
    
    def __call__(self, wav, *args: Any, **kwds: Any) -> Any:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)