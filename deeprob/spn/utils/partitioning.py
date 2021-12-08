# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

from __future__ import annotations
import itertools
from typing import Optional, Tuple

import numpy as np


class Partition:

    def __init__(
        self,
        row_ids: list,
        col_ids: list,
        uncond_vars: list,
        parent_partition: Optional[Partition] = None,
        is_naive: Optional[bool] = False,
        is_conj: Optional[bool] = False,
    ):
        """
        Create a partition, i.e. an object modeling a data slice (and some of its properties)
        by keeping track of its indices (i.e. row_ids and col_ids).

        :param row_ids: The row indices of the modeled slice.
        :param col_ids: The column indices of the modeled slice.
        :param uncond_vars:  Ordered list of variables from which the conjunction variables will be extracted
         to horizontally split the current partition.
        :param parent_partition: The optional parent partition
        :param is_naive: If True and determinism is not required, a naive factorization will be learnt over the data
         slice modeled by the current partition; otherwise, if True and determinism is required, a disjunction
         will be learnt over the data slice modeled by the current partition.
        :param is_conj: True if the modeled slice is associated to a conjunction, i.e. every row in the slice is
         equal to the others.
        """
        self.row_ids = np.array(row_ids)
        self.col_ids = np.array(col_ids)
        self.uncond_vars = list(uncond_vars)

        self.parent_partition = parent_partition
        self.set_parent_partition(parent_partition)
        self.sub_partitions = []

        self.is_naive = is_naive
        self.is_conj = is_conj
        # discarded assignments, see build_leaf() in xpc.py
        self.disc_assignments = None

    def set_parent_partition(self, parent_partition: Partition):
        """
        Set the parent partition and update its sub_partitions attribute.

        :param parent_partition: The parent partition.
        """
        if parent_partition is not None:
            parent_partition.sub_partitions.append(self)

    def is_partitioned(self):
        """
        :return: True if the partition is partitioned, False otherwise.
        """
        return len(self.sub_partitions) != 0

    def is_horizontally_partitioned(self):
        """
        :return: True if the partition is horizontally partitioned, False otherwise.
        """
        ret = False
        if self.is_partitioned():
            ret = len(self.row_ids) > len(self.sub_partitions[0].row_ids)
        return ret

    def get_slice(self, data: np.ndarray) -> np.ndarray:
        """
        Slice the input data matrix according to self.

        :param data: The data to be sliced.
        :return: The data slice.
        """
        return data[self.row_ids][:, self.col_ids]

    def get_vertical_split(self) -> list[np.ndarray, np.ndarray]:
        """
        If possible, split vertically the current partition.
        """
        vertical_split = []
        cond_vars = [col_id for col_id in self.col_ids if col_id not in self.uncond_vars]
        if len(cond_vars) != 0 and len(cond_vars) != len(self.col_ids):
            vertical_split = [np.asarray(cond_vars), np.asarray(self.uncond_vars)]
        return vertical_split

    def get_conj_row_ids(
        self,
        data: np.ndarray,
        conj: list,
        min_part_inst: int,
    ) -> np.ndarray:
        """
        Return the row ids of the instances satisfying the given conjunction.
        The row ids must be found within the slice modeled by the self partition.

        :param data: The input data.
        :param conj: Conjunction modeled as a list of two lists: the first contains
                     the IDs of the variables, the second  the related assignment.
                     For example, [[8,3],[1,0]] models the conjunction X8=1 and X3=0.
        :param min_part_inst: the minimum number of instances allowed to return.
        :return: The row ids of the instances satisfying the given conjunction iff
                 the number of such instances is greater or equal than the minimum number
                 of instances allowed to return; otherwise, an empty array.
        """
        if len(self.row_ids) < min_part_inst:
            conj_row_ids = np.empty(0, dtype=np.int32)
        else:
            conj_row_ids = self.row_ids.copy()
            for i in range(len(conj[0])):
                conj_row_ids = conj_row_ids[np.where(data[np.array(conj_row_ids), conj[0][i]] == conj[1][i])[0]]
                if len(conj_row_ids) < min_part_inst:
                    conj_row_ids = np.empty(0, dtype=np.int32)
                    break
        return conj_row_ids

    def get_horizontal_split(
        self,
        data: np.ndarray,
        min_part_inst: int,
        conj_len: int,
        arity: int,
        sd: bool,
        random_state: np.random.RandomState
    ) -> Tuple[list, np.ndarray, list]:
        """
        If possible, split horizontally the current partition.

        :param data: The input data matrix.
        :param min_part_inst: The minimum number of instances allowed per partition.
        :param conj_len: The conjunction length.
        :param arity: The maximum number of subpartitions for an horizontal partitioned partition.
        :param sd: True if the generated tree will be used to model a SD PC, False otherwise.
        :param random_state: The random state.
        """
        if len(self.uncond_vars) < conj_len or len(self.row_ids) < 2 * min_part_inst:
            return [], np.array([]), []

        uncond_vars = self.uncond_vars.copy()
        if not sd:
            random_state.shuffle(uncond_vars)
        conj_vars = uncond_vars[:conj_len]

        # list of all possible assignments for a conjunction with length conj_len
        assignments = [list(assignment) for assignment in itertools.product([0, 1], repeat=len(conj_vars))]
        random_state.shuffle(assignments)

        discarded_row_ids = self.row_ids.copy()
        conj_row_ids_l = []
        for assignment in assignments:
            conj = [conj_vars, assignment]
            conj_row_ids = self.get_conj_row_ids(data, conj, min_part_inst)
            if len(conj_row_ids) == len(self.row_ids):
                return [], discarded_row_ids, conj_vars
            if len(conj_row_ids) != 0 and len(discarded_row_ids) - len(conj_row_ids) >= min_part_inst:
                discarded_row_ids = np.setdiff1d(discarded_row_ids, conj_row_ids)
                conj_row_ids_l.append(conj_row_ids)
                if len(conj_row_ids_l) == arity - 1 or len(discarded_row_ids) < 2 * min_part_inst:
                    break

        if conj_row_ids_l:
            return conj_row_ids_l, discarded_row_ids, conj_vars
        return [], np.array([]), []


def generate_random_partitioning(
    data: np.ndarray,
    min_part_inst: int,
    n_max_parts: int,
    conj_len: int,
    arity: int,
    sd: bool,
    uncond_vars: list,
    random_state: np.random.RandomState
):
    """
    Create a random partition tree.

    :param data: The input data matrix.
    :param min_part_inst: The minimum number of instances allowed per partition.
    :param n_max_parts: The maximum number of partitions in the tree.
    :param conj_len: The conjunction length.
    :param arity: The maximum number of subpartitions for an horizontal partitioned partition.
    :param sd: True if the generated tree will be used to model a SD PC, False otherwise.
    :param uncond_vars: Ordered list of variables from which the first *conj_len* ones
     are extracted as conjunction variables to partition the root partition.
    :param random_state: The random state.

    :return partition_root: The partition root of the tree.
    :return cl_parts_l: List containing the leaf partitions over which a CLTree will be learnt.
    :return conj_vars_l: List of lists. Every sublist contains the variables of a conjunction (e.g. [[3, 5]]).
     If a sublist occurs before another, then the former has been used first. There are no duplicates.
    :return n_partitions: The number of partitions in the generated tree.
    """
    partition_root = Partition(row_ids=np.arange(data.shape[0]),
                               col_ids=uncond_vars,
                               uncond_vars=uncond_vars,
                               parent_partition=None)
    n_partitions = 0
    conj_vars_l = []
    cl_parts_l = []
    leaves = [partition_root]
    while leaves and n_partitions + len(leaves) < n_max_parts:
        # randomly pop a leaf partition
        part = leaves.pop(random_state.randint(len(leaves)))

        conj_row_ids_l, discarded_row_ids, conj_vars = \
            part.get_horizontal_split(data, min_part_inst, conj_len, arity, sd, random_state)

        if len(discarded_row_ids):
            if conj_vars not in conj_vars_l:
                conj_vars_l.append(conj_vars)

            # this ensures a general definition of the list uncond_vars, preserving its order
            uncond_vars = [uv for uv in part.uncond_vars if uv not in conj_vars]

            part_buffer = [
                Partition(row_ids=discarded_row_ids,
                          col_ids=part.col_ids.copy(),
                          uncond_vars=uncond_vars.copy(),
                          parent_partition=part)]

            for conj_row_ids in conj_row_ids_l:
                part_buffer.append(
                    Partition(row_ids=conj_row_ids,
                              col_ids=part.col_ids.copy(),
                              uncond_vars=uncond_vars.copy(),
                              parent_partition=part))

            discarded_assignments = \
                {tuple(assignment) for assignment in itertools.product([0, 1], repeat=len(conj_vars))}

            for k in range(len(part_buffer)):
                part = part_buffer[k]
                vertical_split = part.get_vertical_split()
                if vertical_split:
                    n_partitions += 1
                    is_conj = False if not k else True
                    p = Partition(row_ids=part.row_ids.copy(),
                                  col_ids=vertical_split[0].copy(),
                                  uncond_vars=[],
                                  parent_partition=part,
                                  is_naive=True,
                                  is_conj=is_conj)
                    if is_conj:
                        discarded_assignments.remove(tuple(p.get_slice(data)[0]))

                    leaves.append(
                        Partition(row_ids=part.row_ids.copy(),
                                  col_ids=vertical_split[1].copy(),
                                  uncond_vars=vertical_split[1].copy(),
                                  parent_partition=part))
                else:
                    leaves.append(part)

            if part_buffer[0].sub_partitions:
                part_buffer[0].sub_partitions[0].disc_assignments = \
                    np.array(list(discarded_assignments))
        else:
            n_partitions += 1
            cl_parts_l.append(part)

    # in case the process ended because n_partitions + len(leaves) > n_max_parts
    n_partitions += len(leaves)
    cl_parts_l.extend(leaves)
    return partition_root, cl_parts_l, conj_vars_l, n_partitions
