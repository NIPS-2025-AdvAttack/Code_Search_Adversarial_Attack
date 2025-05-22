import sys
from tree_sitter import Language, Parser
import tree_sitter_go

def parse_go(go_code_text):
    output = {
        "variables": {},
        "func_names": {}
    }
    
    go_code = go_code_text.encode('utf8')

    # set up parser
    GO_LANGUAGE = Language(tree_sitter_go.language())
    parser = Parser(GO_LANGUAGE)

    tree = parser.parse(go_code)
    root = tree.root_node

    # collectors
    imports = set()
    funcs   = []  # function declarations
    vars_   = []  # short/var declarations
    params  = []  # parameter declarations
    usages  = []  # all other identifiers

    def walk(node):
        # print("Node: ", node)
        # ─── imports ────────────────────────────────────────────────
        if node.type == "import_spec":
            alias = node.child_by_field_name("name")
            if alias and alias.type == "identifier":
                imports.add(get_text(alias))
            else:
                path = node.child_by_field_name("path")
                pkg  = get_text(path)[1:-1].split("/")[-1]  # strip quotes
                imports.add(pkg)
            return  # don’t recurse into the string literal

        # ─── function name ────────────────────────────────────────
        if node.type == "function_declaration":
            name = node.child_by_field_name("name")
            funcs.append(name)

        # ─── parameters ────────────────────────────────────────────
        # tree-sitter-go calls these "parameter_declaration"
        elif node.type == "parameter_declaration":
            # there may be N identifiers before a type, e.g. (a, b int)
            for child in node.children:
                if child.type == "identifier":
                    params.append(child)
            # no need to recurse into its children—they’re just identifiers + type

        # ─── short var (:=) ────────────────────────────────────────
        elif node.type == "short_var_declaration":
            left = node.child_by_field_name("left")
            for id_node in left.children:
                if id_node.type == "identifier":
                    vars_.append(id_node)

        # ─── var keyword ───────────────────────────────────────────
        elif node.type == "var_spec":
            for child in node.children:
                if child.type == "identifier":
                    vars_.append(child)

        # ─── all other identifiers ──────────────────────────────────
        elif node.type == "identifier":
            name = get_text(node)
            # ignore imports, function names & declarations themselves
            if name not in imports:
                usages.append(node)
        
        elif node.type == "range_clause":
            # range clauses are like for loops
            # e.g. for i, v := range x
            # we want to record the i and v identifiers
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left:
                for id_node in left.children:
                    if id_node.type == "identifier":
                        vars_.append(id_node)
            if right:
                for id_node in right.children:
                    if id_node.type == "identifier":
                        vars_.append(id_node)

        # recurse
        for c in node.children:
            walk(c)

    def get_text(node):
        return go_code[node.start_byte:node.end_byte].decode('utf8')

    walk(root)

    # build a map: name → list of (kind, line, col)
    occurrences = {}

    def record(nodes, kind):
        for n in nodes:
            name = get_text(n)
            occurrences.setdefault(name, []).append((
                kind,
                n.start_point[0] + 1,
                n.start_point[1] + 1,
                n.start_byte,
                n.end_byte,
            ))

    # record all declaration kinds
    record(funcs,  "func_decl")
    record(vars_,  "var_decl")
    record(params, "param_decl")
    # record usages, but skip any whose location we’ve already recorded
    for n in usages:
        name = get_text(n)
        if name not in occurrences:
            continue
        # compute 1-based line/col
        line, col = n.start_point[0] + 1, n.start_point[1] + 1
        start_offset = n.start_byte
        end_offset = n.end_byte

        # collect all already-recorded positions for this name
        existing_positions = {
            (l, c)
            for _, l, c, _, _ in occurrences[name]
        }

        # only append if this (line,col) is brand new
        if (line, col) not in existing_positions:
            occurrences[name].append(("usage", line, col, start_offset, end_offset))


    # ─── print a per-name summary ────────────────────────────────
    # for name, occs in sorted(occurrences.items()):
    #     print(f"\n{name!r}:")
    #     for kind, line, col, _, _ in occs:
    #         print(f"  • {kind:10s} at line {line}, col {col}")
            
    def deduplicate_occurences(occs):
        """Remove duplicate occurrences from the list."""
        seen = set()
        deduped = []
        for kind, line, col, start_offset, end_offset in occs:
            if (line, col) not in seen:
                seen.add((line, col))
                deduped.append((kind, line, col, start_offset, end_offset))
        return deduped

    for name, occs in sorted(occurrences.items()):
        # Turn each (kind, line, col) into your dict
        entries = [
            {"loc": [start_offset, end_offset], "text": name}
            for kind, line, col, start_offset, end_offset in deduplicate_occurences(occs)
        ]

        # If at least one occurrence is a function declaration, treat it as func_names
        if any(kind == "function_decl" for kind, _, _, _, _ in deduplicate_occurences(occs)):
            output["func_names"].setdefault(name, []).extend(entries)
        else:
            # Otherwise treat it as a variable
            output["variables"].setdefault(name, []).extend(entries)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/file.go")
        sys.exit(1)
    fname = sys.argv[1]
    if not fname.endswith('.go'):
        print("Warning: file does not end in .go")

    with open(fname, 'r') as f:
        go_code = f.read()
    parse_go(go_code)
