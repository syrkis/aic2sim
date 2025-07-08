# %% dsl.py
#   domain specific language
# by: Noah Syrkis

# Imports
from functools import reduce

import jax.numpy as jnp
from jax import tree
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from nebellum.types import Tree
from nebellum.act import a2i


# %% Grammar
grammar = Grammar(r"""
tree        = node (sep node)*
node        = fallback / sequence / action / condition

fallback    = "F" ws "(" ws tree ws ")" ws
sequence    = "S" ws "(" ws tree ws ")" ws
action      = "A" ws (move / stand / shoot) ws
condition   = "C" ws (in_reach / in_sight) ws

in_reach    = "in_reach" ws team
in_sight    = "in_sight" ws team
move        = "move" ws target
shoot       = "shoot" ws qualifier
stand       = "stand"
target      = "target"
qualifier   = "random"
team        = "ally" / "enemy"

sep         = ws "|>" ws
ws          = ~r"\s*"
""")


# %% Functions
def bts_fn(bt_strs) -> Tree:
    bts = tuple(map(lambda x: txt2bts(x.strip()), bt_strs.strip().split("\n")))  # type: ignore
    bts = tree.map(lambda *x: jnp.stack(x), *bts)  # type: ignore
    return bts


def txt2bts(txt) -> Tree:
    struct = BehaviorTreeVisitor().visit(grammar.parse(txt))
    fns = [idxs_fn, parent_fn, skips_fn, prevs_fn]
    idxs, over, jump, left = map(jnp.array, [f(struct) for f in fns])
    bt = Tree(idxs=idxs, over=over, jump=jump, left=left)
    return tree.map(lambda x: jnp.pad(x, (0, len(a2i) - x.size)), bt)


# %% Tree traversales
def skips_fn(node):
    def aux_fn(n):
        if "children" not in n:
            return 1
        return sum(aux_fn(child) for child in n["children"])  # type: ignore

    if "children" not in node:
        return [0]  # No leaves after a leaf

    result = []
    total_leaves = aux_fn(node)
    for i, child in enumerate(node["children"]):
        child_leaves = aux_fn(child)
        leaves_after = total_leaves - sum(aux_fn(node["children"][j]) for j in range(i + 1))
        result.extend([leaves_after] * child_leaves if "children" not in child else skips_fn(child))
    return result


def prevs_fn(node):
    """
    Returns a list of length n_leafs. Each entry is 0 (for fallback) or 1 (for sequence).
    The entry indicates the parent type of the immediately preceding leaf.
    For the first leaf in a sequence of siblings, it gets the parent's type.
    For subsequent leaves, they get the type of their parent.
    """

    def traverse(n, parent_type):
        if n.get("type") in ["condition", "action"]:
            return [parent_type]  # Leaf node

        node_type = n["type"] == "sequence"  # 0 for fallback, 1 for sequence
        result = []
        for i, child in enumerate(n["children"]):
            # For first child, use parent's type, for others use current node's type
            prev_type = parent_type if i == 0 else node_type
            if child.get("type") in ["condition", "action"]:
                result.append(prev_type)
            else:
                result.extend(traverse(child, prev_type))
        return result

    # Start with 0 as default parent type for the root (could be either 0 or root's type)
    return traverse(node, 0)


def idxs_fn(node):  # CORRECT
    if "children" not in node:
        node = node[node.get("type")]
        return [a2i[(node,)] if type(node) is str else a2i[tuple(node.values())]]  # type: ignore
    else:
        return reduce(lambda acc, child: acc + idxs_fn(child), node["children"], [])


def parent_fn(node):  # CORRECT
    parents = []
    if "children" in node:
        for child in node["children"]:
            if "children" not in child:
                parents += [int(node["type"] == "sequence")]
            else:
                parents += parent_fn(child)
    else:
        parents += [int(node["type"] == "sequence")]
    return parents


# %% Visitor (parsimonious stuff) BELOW HERE THERE BE DRAGONS
class BehaviorTreeVisitor(NodeVisitor):
    def visit_tree(self, node, visited_children):
        """Process the full tree with all its nodes."""
        first_node, rest = visited_children

        if rest:
            # Start with the first node
            nodes = [first_node]

            # For each item in rest
            for item in rest:
                # The separator is at index 0 (a string)
                # The node is at index 1 (a dictionary)
                if isinstance(item, list) and len(item) > 1:
                    nodes.append(item[1])
                elif type(item) is str and item.startswith("|>"):
                    continue
                else:
                    nodes.append(item)

            return nodes
        return first_node

    def visit_node(self, node, visited_children):
        """Process a node, which can be fallback, sequence, action or condition."""
        # Just return the first (and only) child
        return visited_children[0]

    def visit_fallback(self, node, visited_children):
        """Process a fallback node (F)."""
        # The structure is ["F", ws, "(", ws, tree, ws, ")", ws]
        return {"type": "fallback", "children": visited_children[4]}

    def visit_sequence(self, node, visited_children):
        """Process a sequence node (S)."""
        # The structure is ["S", ws, "(", ws, tree, ws, ")", ws]
        _, _, _, _, tree, *_ = visited_children
        return {"type": "sequence", "children": tree}

    def visit_action(self, node, visited_children):
        """Process an action node (A)."""
        # The structure is ["A", ws, move_or_stand, ws]
        _, _, action_data, *_ = visited_children
        return {"type": "action", "action": action_data}

    def visit_condition(self, node, visited_children):
        """Process a condition node (C)."""
        # The structure is ["C", ws, condition, ws]
        _, _, condition, *_ = visited_children
        return {"type": "condition", "condition": condition}

    def visit_move(self, node, visited_children):
        """Process a move action."""
        # The structure is ["move", ws, target]
        _, _, target, *_ = visited_children
        return {"name": "move", "target": target}

    def visit_shoot(self, node, visited_children):
        """Process a shoot action."""
        # The structure is ["shoot", ws, qualifier]
        _, _, qualifier, *_ = visited_children
        return {"name": "shoot", "qualifier": qualifier}

    def visit_stand(self, node, visited_children):
        """Process a stand action."""
        return {"name": "stand"}

    def visit_target(self, node, visited_children):
        """Process the target."""
        return node.text

    def visit_qualifier(self, node, visited_children):
        """Process the qualifier."""
        return node.text

    def visit_in_reach(self, node, visited_children):
        """Process an in_reach condition."""
        # The structure is ["in_reach", ws, team]
        _, _, team, *_ = visited_children
        return {"name": "in_reach", "team": team}

    def visit_in_sight(self, node, visited_children):
        """Process an in_sight condition."""
        # The structure is ["in_sight", ws, team]
        _, _, team, *_ = visited_children
        return {"name": "in_sight", "team": team}

    def visit_team(self, node, visited_children):
        """Process the team."""
        return node.text

    # Handle whitespace and separators (usually just returning them or ignoring them)
    def visit_sep(self, node, visited_children):
        """Process a separator."""
        return node.text

    def visit_ws(self, node, visited_children):
        """Process whitespace."""
        return node.text

    # Generic visit method for any other nodes
    def generic_visit(self, node, visited_children):
        """The generic visit method."""
        if visited_children and len(visited_children) == 1:
            return visited_children[0]
        return visited_children or node.text
