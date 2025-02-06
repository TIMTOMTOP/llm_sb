import ast
import astor
import os
import networkx as nx
from typing import Optional
import anthropic

class SDKAnalyzer:
    def __init__(self, sdk_path: str):
        self.sdk_path = sdk_path
        self.graph = nx.DiGraph()

    ##############################
    #          First Pass        #
    ##############################
    def first_pass(self):
        """
        Walk through the SDK directory and process each Python file.
        """
        for root, _, files in os.walk(self.sdk_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self._first_pass_file(filepath)

    def _first_pass_file(self, filepath: str):
        """
        Parse a single file and add nodes for:
          - the file,
          - any classes (and their methods),
          - and any top-level functions.
          
        Each node is added with only two attributes: 'type' and 'name'.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code)

            # Add file node (store only type and name)
            file_node_id = f"File:{filepath}"
            self.graph.add_node(file_node_id, type='file', name=os.path.basename(filepath))

            # Use a set to track methods that are defined inside classes (to avoid re-adding them as top-level functions)
            method_nodes = set()

            for node in ast.walk(tree):
                # Process classes
                if isinstance(node, ast.ClassDef):
                    class_node_id = f"Class:{node.name}@{filepath}"
                    self.graph.add_node(class_node_id, type='class', name=node.name)
                    self.graph.add_edge(file_node_id, class_node_id, relationship='contains')

                    # Process methods inside the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_node_id = f"Method:{node.name}.{item.name}@{filepath}"
                            self.graph.add_node(method_node_id, type='method', name=item.name)
                            self.graph.add_edge(class_node_id, method_node_id, relationship='contains')
                            method_nodes.add(f"{filepath}:{item.name}")

                # Process top-level functions
                elif isinstance(node, ast.FunctionDef) and f"{filepath}:{node.name}" not in method_nodes:
                    func_node_id = f"Function:{node.name}@{filepath}"
                    self.graph.add_node(func_node_id, type='function', name=node.name)
                    self.graph.add_edge(file_node_id, func_node_id, relationship='contains')

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    ##############################
    #         Second Pass        #
    ##############################
    def second_pass(self):
        """
        Walk through each file node (using its file path from its node id) and re-parse the file
        to establish relationships (calls and inheritance). External nodes are also added with only
        'type' and 'name' attributes.
        """
        for node_id, data in list(self.graph.nodes(data=True)):
            if data.get('type') == 'file':
                # Extract the full file path from the node id (formatted as "File:<filepath>")
                try:
                    filepath = node_id.split(":", 1)[1]
                    if os.path.isfile(filepath):
                        self._second_pass_file(filepath)
                except Exception as ex:
                    print(f"Error extracting filepath from {node_id}: {ex}")

    def _second_pass_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            self.populate_parents(tree)

            # Process function/method calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    callee_name = None
                    if isinstance(node.func, ast.Name):
                        callee_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        callee_name = node.func.attr
                    if callee_name:
                        caller_ast = self._get_enclosing_function_or_method(node)
                        if caller_ast:
                            caller_id = self._make_node_id_for(caller_ast,filepath)
                            if caller_id and caller_id in self.graph:
                                callee_id = self._get_function_global_by_name(callee_name)
                                if callee_id is None:
                                    callee_id = self._get_class_global_by_name(callee_name)
                                if callee_id is None:
                                    callee_id = f"Function:{callee_name}@/"
                                    if callee_id not in self.graph:
                                        self.graph.add_node(callee_id,
                                                            type='external_function',
                                                            name=callee_name)
                                if callee_id and callee_id in self.graph:  # Additional check for callee
                                    self.graph.add_edge(caller_id, callee_id, relationship='calls')

            # Process inheritance relationships
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.bases:
                    this_class_id = self._make_class_node_id(node, filepath)
                    for base_expr in node.bases:
                        if isinstance(base_expr, ast.Name):
                            base_name = base_expr.id
                            base_id = self._get_class_global_by_name(base_name)
                            if base_id is None:
                                base_id = f"Class:{base_name}@/"
                                if base_id not in self.graph:
                                    self.graph.add_node(base_id,
                                                        type='external_class',
                                                        name=base_name)
                            self.graph.add_edge(this_class_id, base_id, relationship='inherits')

        except Exception as e:
            print(f"Error in second pass for {filepath}: {e}")


    ##############################
    #       Helper Methods       #
    ##############################
    def populate_parents(self, node: ast.AST, parent: Optional[ast.AST] = None):
        """
        Recursively assign a temporary _ast_parent attribute to each node.
        """
        for child in ast.iter_child_nodes(node):
            child._ast_parent = node
            self.populate_parents(child, node)

    def _get_enclosing_function_or_method(self, node: ast.AST) -> Optional[ast.FunctionDef]:
        """
        Traverse upward from the given node until a FunctionDef is found.
        """
        current = node
        while current:
            current = self._get_parent(current)
            if isinstance(current, ast.FunctionDef):
                return current
        return None

    def _get_function_global_by_name(self, name: str) -> Optional[str]:
        """
        Find a node whose type is either 'function' or 'method' with the given name.
        """
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') in ('function', 'method') and data.get('name') == name:
                return node_id
        return None

    def _get_class_global_by_name(self, name: str) -> Optional[str]:
        """
        Find a node whose type is 'class' with the given name.
        """
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'class' and data.get('name') == name:
                return node_id
        return None

    def _make_node_id_for(self, node: ast.AST, filepath: str) -> Optional[str]:
        """
        Reconstruct the node id (for a function or method) from an AST node.
        """
        func_name = None
        cls_name = None

        if isinstance(node, ast.FunctionDef):
            func_name = node.name

        parent = node
        while parent:
            parent = self._get_parent(parent)
            if isinstance(parent, ast.ClassDef):
                cls_name = parent.name
            if isinstance(parent, ast.FunctionDef):
                func_name = parent.name

        if func_name and cls_name:
            return f"Method:{cls_name}.{func_name}@{filepath}"
        elif func_name:
            return f"Function:{func_name}@{filepath}"
        return None

    def _make_class_node_id(self, node: ast.ClassDef, filepath: str) -> str:
        return f"Class:{node.name}@{filepath}"

    def _get_parent(self, node: ast.AST) -> Optional[ast.AST]:
        return getattr(node, '_ast_parent', None)

    def get_node_source(self, node: ast.AST) -> str:
        try:
            return astor.to_source(node).strip()
        except Exception:
            return ""

    def parse_sdk_directory(self):
        """
        Execute the first and second passes on the SDK codebase.
        """
        self.first_pass()
        self.second_pass()

    ##############################
    #      Saving & Loading      #
    ##############################
    def save_graph(self, output_path: str, format: str = 'gexf'):
        """
        Save the graph to a file in the specified format.
        Before saving, any extra node attributes are removed so that each node only has 'type' and 'name'.
        """
        if format in ['gexf', 'graphml']:
            save_graph = self.graph.copy()
            for node, data in save_graph.nodes(data=True):
                # Remove any keys other than 'type' and 'name'
                for key in list(data.keys()):
                    if key not in ('type', 'name'):
                        del data[key]
            if format == 'gexf':
                nx.write_gexf(save_graph, output_path)
            else:
                nx.write_graphml(save_graph, output_path)
        elif format == 'pkl':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            raise ValueError("Format must be 'gexf', 'graphml', or 'pkl'.")

    def load_graph(self, input_path: str, format: str = 'gexf'):
        if format == 'gexf':
            self.graph = nx.read_gexf(input_path)
        elif format == 'graphml':
            self.graph = nx.read_graphml(input_path)
        elif format == 'pkl':
            import pickle
            with open(input_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            raise ValueError("Format must be 'gexf', 'graphml', or 'pkl'.")


########################################
#  Example usage:
########################################

if __name__ == '__main__':
    # For example, using the anthropic package's directory as the SDK
    sdk_path = os.path.dirname(anthropic.__file__)
    analyzer = SDKAnalyzer(sdk_path)
    analyzer.parse_sdk_directory()  # runs both passes
    analyzer.save_graph("sdk_graph_anthropic.pkl", format="pkl")
    analyzer.save_graph("sdk_graph_anthropic.graphml", format="graphml")
