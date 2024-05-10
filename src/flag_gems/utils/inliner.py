from typing import List
import ast

from triton.runtime import JITFunction
from flag_gems.utils.code_utils import NameSpace


class Rename(ast.NodeTransformer):
    """Rename references to variables"""
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_arg(self, node):
        if node.arg in self.mapping:
            return ast.arg(**{**node.__dict__, 'arg': self.mapping[node.arg]})
        return node

    def visit_Name(self, node):
        if node.id in self.mapping:
            return ast.Name(**{**node.__dict__, 'id': self.mapping[node.id]})
        return node


class HandleReturn(ast.NodeTransformer):
    """Replace return statements with assignments"""
    def __init__(self, return_names):
        self.return_names = return_names

    def visit_Return(self, node: ast.Return):
        if isinstance(node.value, ast.Tuple):
            targets = ast.Tuple(
                elts=tuple(
                    ast.Name(id=self.return_names[i], ctx=ast.Store())
                    for i in range(len(self.return_names))),
                ctx=ast.Store(),
            )
        else:
            targets = ast.Name(id=self.return_names[0])  # only one return
        new_node = ast.Assign([targets], value=node.value)
        ast.copy_location(new_node, node)
        return new_node


class NameCollector(ast.NodeVisitor):
    """Collect all assignemnts to variables in a function"""
    def __init__(self):
        self.names = set()

    def visit_Assign(self, node):
        for target in node.targets:
            self.names.add(target.id)

    def visit_AugAssign(self, node):
        self.names.add(node.target.id)

    def visit_AnnAssign(self, node):
        self.names.add(node.target.id)


def inline_function(f: JITFunction, input_names: List[str],
                    output_names: List[str], namespace: NameSpace):
    nc = NameCollector()
    ast_tree = ast.parse(f.src)
    nc.visit(ast_tree)

    mapping = {}
    # replace arguemnt names with input names
    # TODO: handle non-tensor arguments to scalar function
    for p, input_name in zip(f.arg_names, input_names):
        mapping[p] = input_name

    # replace temporaries with auto-named names to avoid conflicts
    for name in nc.names:
        mapping[name] = namespace.create_name(name)

    rename = Rename(mapping)
    renamed_tree = rename.visit(ast_tree)

    # replace return statements with assignments to output names
    return_ = HandleReturn(output_names)
    renamed_tree = return_.visit(renamed_tree)

    body = renamed_tree.body[0].body
    return ast.unparse(body)
