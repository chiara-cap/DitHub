import torch

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class TaskMemory:
    def __init__(self):
        self._current_classes = []
        self._modules = {}
        self._memory = {}
        """
        self._modules = {
            layer_id: {
                class_name: [tensor_A, tensor_B]
            }
        }
        """
        self._registered_instances = []
        self._class_counter = {}
        self.current_task = None
        self.is_warm_up = True
        self.task_index = 0
        self.task_mapping = None

    def get_classes(self):
        assert self._current_classes, 'Please set self._current_classes before getting them'
        return self._current_classes

    def set_classes(self, classes):
        self._current_classes = classes

    def get_modules(self, layer_id):
        if layer_id not in self._modules:
            return {}
        return self._modules[layer_id]

    def set_counter(self, classes):
        if not self._class_counter:
            for c in classes:
                self._class_counter[c] = 0

    def add_counter(self, classes):
        if self.is_warm_up:
            return
        for c in classes:
            self._class_counter[c[6:]] += 1

    def print_counter(self):
        for key in sorted(self._class_counter.keys()):
            print(f"{key}: {self._class_counter[key]}")

    @torch.no_grad()
    def save_modules(self):
        for m in self._registered_instances:
            if m.layer_name not in self._modules:
                self._modules[m.layer_name] = {}
            curr_classes = m.per_class_lora_A.keys()
            for curr_class in curr_classes:
                a_b = [
                    m.per_class_lora_A[curr_class]
                ]
                self._modules[m.layer_name][curr_class] = a_b
                self.set_memory("B_shared", m.layer_name, m.shared_lora_b)
        self._registered_instances = []

    @torch.no_grad()
    def merge_B(self):
        if any(k.startswith("B_prev_task") for k in self._memory):
            lambda_b_marouf = 0.7
            for key in self._memory.keys():
                if key.startswith("B_shared"):
                    suffix = key[len("B_shared"):]
                    prev_task_key = "B_prev_task" + suffix
                    assert prev_task_key in self._memory, f"{prev_task_key} non trovato in self._memory"
                    self._memory[key] = (1 - lambda_b_marouf) * self._memory[prev_task_key] + lambda_b_marouf * self._memory[key]

    def set_memory(self, key, layer_name, value):
        assert isinstance(value, torch.Tensor), f"Value must be a tensor, but got {type(value)}"
        self._memory[key+layer_name] = value

    def get_memory(self, key, layer_name):
        return self._memory.get(key+layer_name, None)

    def end_task(self):
        self.save_modules()
        self.print_counter()
        self.merge_B()
        self.is_warm_up = True
        self.task_index += 1

    def enable_per_class(self):
        self.is_warm_up = False
        for m in self._registered_instances:
            m.enable_per_class(self.current_task)

    def register_module(self, instance):
        self._registered_instances.append(instance)
