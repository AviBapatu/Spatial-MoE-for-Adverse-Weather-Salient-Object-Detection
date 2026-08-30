import ast

code = """
relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
    self.window_size * self.window_size, self.window_size * self.window_size, -1)
"""

print(ast.dump(ast.parse(code), indent=2))
