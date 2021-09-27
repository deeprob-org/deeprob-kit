import unittest
import numpy as np

from deeprob.utils.graph import build_tree_structure, compute_bfs_ordering, maximum_spanning_tree


class TestGraph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGraph, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.tree = [1, -1, 0, 0, 1, 4, 4, 5]
        cls.scope = [3, 5, 2, 0, 7, 1, 4, 9]
        cls.adj_matrix = np.array([
            [0, 8, 0, 3],
            [0, 0, 2, 5],
            [0, 0, 0, 6],
            [0, 0, 0, 0]
        ])

    def test_build_tree_structure(self):
        root = build_tree_structure(self.tree, self.scope)
        node_ids = [root.get_id()]
        node_ids += [c.get_id() for c in root.get_children()]
        node_ids += [d.get_id() for c in root.get_children() for d in c.get_children()]
        node_ids += [root.get_children()[1].get_children()[0].get_children()[0].get_id()]
        tree, scope = root.get_tree_scope()
        self.assertEqual(node_ids, [5, 3, 7, 2, 0, 1, 4, 9])
        self.assertEqual(root.get_n_nodes(), 8)
        self.assertEqual(set(scope), set(self.scope))
        self.assertEqual(
            set(map(lambda x: (-1, x[1]) if x[0] == -1 else (scope[x[0]], x[1]), zip(tree, scope))),
            set(map(lambda x: (-1, x[1]) if x[0] == -1 else (self.scope[x[0]], x[1]), zip(self.tree, self.scope)))
        )
        self.assertRaises(ValueError, build_tree_structure, [0, 0, 1, 2, 3])
        self.assertRaises(ValueError, build_tree_structure, [-1, 0, -1, 2, 3])

    def test_compute_bfs_ordering(self):
        ordering = compute_bfs_ordering(self.tree)
        self.assertEqual(ordering, [1, 0, 4, 2, 3, 5, 6, 7])

    def test_maximum_spanning_tree(self):
        bfs, tree = maximum_spanning_tree(0, self.adj_matrix)
        self.assertEqual(bfs.tolist(), [0, 1, 3, 2])
        self.assertEqual(tree.tolist(), [-1, 0, 3, 1])


if __name__ == '__main__':
    unittest.main()
