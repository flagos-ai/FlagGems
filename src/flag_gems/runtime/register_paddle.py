import sys


class RegisterPaddle:
    def __init__(self, config, user_unused_ops_list=None):
        """
        Paddle operator registrar

        Args:
            config: List of operator configurations, format: [(op_name, custom_op), ...]
                   where op_name is the operator name and custom_op is the custom operator function
            user_unused_ops_list: List of operators specified by user to not use
        """
        self.all_ops = []
        self.unused_ops = user_unused_ops_list or []
        self.config = config
        self.original_ops = {}
        self.config_filter()
        self.for_each()

    def config_filter(self):
        self.config = [
            item for item in self.config if item[1].__name__ not in self.unused_ops
        ]

    def register_impl(self, op_name, custom_op):
        import paddle

        if "." in op_name:
            parts = op_name.split(".")
            module_path = parts[:-1]
            attr_name = parts[-1]

            module = paddle
            for part in module_path:
                module = getattr(module, part)

            original_op = getattr(module, attr_name)
            self.original_ops[op_name] = (module, attr_name, original_op)
            setattr(module, attr_name, custom_op)
        else:
            original_op = getattr(paddle, op_name)
            self.original_ops[op_name] = (paddle, op_name, original_op)
            setattr(paddle, op_name, custom_op)
            # The torch proxy copies paddle's namespace when `import torch` runs,
            # so torch.<op> still holds the original function; patch it as well.
            torch_proxy = sys.modules.get("torch")
            if (
                torch_proxy is not None
                and torch_proxy is not paddle
                and torch_proxy.__dict__.get(op_name) is original_op
            ):
                self.original_ops[f"torch.{op_name}"] = (
                    torch_proxy,
                    op_name,
                    original_op,
                )
                setattr(torch_proxy, op_name, custom_op)

        self.all_ops.append(op_name)

    def for_each(self):
        try:
            for item in self.config:
                if len(item) == 2:
                    op_name, custom_op = item
                    self.register_impl(op_name, custom_op)
                else:
                    print(
                        f"Invalid config format: {item}. Expected (op_name, custom_op)"
                    )
        except Exception as e:
            print(f"Error registering paddle ops: {e}")

    def restore_all(self):
        for op_name, saved_info in self.original_ops.items():
            module, attr_name, original_op = saved_info
            setattr(module, attr_name, original_op)

    def restore_op(self, op_name):
        if op_name in self.original_ops:
            module, attr_name, original_op = self.original_ops[op_name]
            setattr(module, attr_name, original_op)
            print(f"Restored paddle op: {op_name}")
        else:
            print(f"Original op not found: {op_name}")

    def get_all_ops(self):
        return self.all_ops

    def get_unused_ops(self):
        return self.unused_ops
