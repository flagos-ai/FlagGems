import argparse
import ast
import logging


class DecoratorFinder(ast.NodeVisitor):
    def __init__(self, target_decorator):
        self.target_decorator = target_decorator
        self.functions_with_decorator = []

    def visit_FunctionDef(self, node):
        # Check if the function has any decorator with the specified name
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if (
                    isinstance(decorator.value, ast.Name)
                    and decorator.value.id == self.target_decorator[0]
                    and decorator.attr == self.target_decorator[1]
                ):
                    self.functions_with_decorator.append(
                        {
                            "name": node.name,
                            "start_line": node.lineno,
                            "end_line": self.get_end_line(node),
                            "body": ast.unparse(node),
                        }
                    )
                    logging.debug(ast.dump(decorator, annotate_fields=True, indent=4))

            elif isinstance(decorator, ast.Call):
                if (
                    isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == self.target_decorator[0]
                    and decorator.func.attr == self.target_decorator[1]
                ):
                    self.functions_with_decorator.append(
                        {
                            "name": node.name,
                            "start_line": node.lineno,
                            "end_line": self.get_end_line(node),
                            "body": ast.unparse(node),
                        }
                    )
                    logging.debug(ast.dump(decorator, annotate_fields=True, indent=4))
        self.generic_visit(node)

    def get_end_line(self, node):
        end_line = node.lineno
        for stmt in node.body:
            if hasattr(stmt, "lineno"):
                end_line = max(
                    end_line,
                    stmt.lineno
                    + (
                        stmt.end_lineno - stmt.lineno
                        if hasattr(stmt, "end_lineno")
                        else 0
                    ),
                )
        return end_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a list of arguments.")
    parser.add_argument(
        "files", nargs="+", help="List of files or arguments to process"
    )
    args = parser.parse_args()

    for filename in args.files:
        if not filename.endswith(".py"):
            continue
        logging.debug(filename)
        with open(filename, "r") as file:
            source_code = file.read()

        tree = ast.parse(source_code)
        decorator_name = "triton.jit"
        target_decorator = decorator_name.split(".")

        finder = DecoratorFinder(target_decorator)
        finder.visit(tree)
        if len(finder.functions_with_decorator) != 0:
            print(filename)
            for item in finder.functions_with_decorator:
                logging.debug(item)
                logging.debug(
                    f"{filename} {item['name']}  {item['start_line']}:{item['end_line']}"
                )
                print(f"--- {item['start_line']} : {item['end_line']}")
