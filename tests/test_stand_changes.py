"""巡演曲目单谱架换册核心的单元测试。"""

from __future__ import annotations

import random
from functools import lru_cache

import pytest

from app.stand_changes import (
    MAX_SHOWS,
    MAX_SLOTS,
    MIN_SLOTS,
    StandChangeError,
    plan_stand_changes,
)


def _brute_force_min_loads(ids, slots_count):
    """状态 DP 参考实现：不计任何换册规则，直接求最少上架次数。

    状态为架上方案集合（frozenset），每场所需册不在架上时任选空槽上架或
    撤一册上架，取总上架次数最小值。与核心实现的"最晚再用"策略相互独立。
    """
    n = len(ids)

    @lru_cache(maxsize=None)
    def dp(i, stand):
        if i == n:
            return 0
        pid = ids[i]
        if pid in stand:
            return dp(i + 1, stand)
        if len(stand) < slots_count:
            return 1 + dp(i + 1, stand | {pid})
        return min(
            1 + dp(i + 1, (stand - {victim}) | {pid}) for victim in stand
        )

    return dp(0, frozenset())


def _check_invariants(result, ids, slots_count):
    """换册结果的结构与规则不变量检查。"""
    assert result.stand_slots == slots_count
    assert result.show_count == len(ids)
    assert len(result.shows) == len(ids)
    assert result.total_loads >= 0

    prev_slots = [None] * slots_count
    loads = 0
    for idx, show in enumerate(result.shows):
        assert show.show_index == idx + 1
        assert show.plan_id == ids[idx]
        assert len(show.slots_after) == slots_count
        # 同一方案至多占一个槽位
        on_stand = [pid for pid in show.slots_after if pid is not None]
        assert len(on_stand) == len(set(on_stand))
        # 每场所需册必须在架上
        assert show.plan_id in on_stand

        if show.load is None:
            # 无需上架：架面不变、无撤下
            assert show.evict is None
            assert show.plan_id in prev_slots
            assert show.slots_after == prev_slots
        else:
            loads += 1
            assert show.load.plan_id == show.plan_id
            assert 1 <= show.load.slot <= slots_count
            assert show.plan_id not in [p for p in prev_slots if p is not None]
            expected = list(prev_slots)
            if show.evict is None:
                # 有空槽才能不撤册，且用编号最小的空槽
                to_free = [i for i, p in enumerate(prev_slots) if p is None]
                assert to_free
                assert show.load.slot == to_free[0] + 1
            else:
                # 撤下与上架同槽：新册入刚腾出的槽位
                assert show.evict.slot == show.load.slot
                assert prev_slots[show.evict.slot - 1] == show.evict.plan_id
                assert show.evict.plan_id not in on_stand
                expected[show.evict.slot - 1] = None
            expected[show.load.slot - 1] = show.plan_id
            assert show.slots_after == expected
        prev_slots = show.slots_after

    assert result.total_loads == loads


@pytest.mark.parametrize("seed", range(300))
def test_random_matches_brute_force_min_loads(seed):
    """随机曲目单：总上架次数与状态 DP 参考实现的最优值一致。"""
    rng = random.Random(seed)
    show_count = rng.randint(1, 14)
    distinct = rng.randint(1, 6)
    ids_pool = [f"p{i:02d}" for i in range(distinct)]
    ids = [rng.choice(ids_pool) for _ in range(show_count)]
    slots_count = rng.randint(MIN_SLOTS, 4)

    result = plan_stand_changes(ids, slots_count)
    assert result.total_loads == _brute_force_min_loads(ids, slots_count)
    _check_invariants(result, ids, slots_count)


@pytest.mark.parametrize("seed", range(100))
def test_random_full_scale_invariants(seed):
    """较大规模（不穷举）下校验结构不变量与最少性必要条件。"""
    rng = random.Random(20_000 + seed)
    show_count = rng.randint(50, MAX_SHOWS)
    distinct = rng.randint(1, 25)
    ids_pool = [f"plan-{i:03d}" for i in range(distinct)]
    ids = [rng.choice(ids_pool) for _ in range(show_count)]
    slots_count = rng.randint(MIN_SLOTS, MAX_SLOTS)

    result = plan_stand_changes(ids, slots_count)
    _check_invariants(result, ids, slots_count)
    # 最少性必要条件：不少于不同方案数，不超过场次数
    assert len(set(ids)) <= result.total_loads <= show_count
    # 槽位不少于不同方案数时，每册恰上架一次、无撤下
    if slots_count >= distinct:
        assert result.total_loads == distinct
        assert all(show.evict is None for show in result.shows)


def test_empty_slots_used_in_number_order():
    result = plan_stand_changes(["a", "b", "c"], 3)
    assert [show.load.slot for show in result.shows] == [1, 2, 3]
    assert result.shows[-1].slots_after == ["a", "b", "c"]
    assert result.total_loads == 3


def test_evict_farthest_next_use():
    # 槽满时撤下未来最晚再次使用的册：b 在第 5 场才再用，a 第 4 场即用
    ids = ["a", "b", "c", "a", "b"]
    result = plan_stand_changes(ids, 2)
    third = result.shows[2]
    assert third.evict is not None
    assert third.evict.plan_id == "b"
    assert third.evict.slot == 2
    assert third.load.slot == 2
    assert third.slots_after == ["a", "c"]
    # 第 4 场 a 已在架上无需动作；第 5 场 b 重新上架
    assert result.shows[3].load is None
    assert result.shows[4].load is not None
    assert result.total_loads == 4


def test_never_used_again_counts_as_farthest():
    # a 之后还会再用、b 永不再用：撤下 b
    ids = ["a", "b", "c", "a"]
    result = plan_stand_changes(ids, 2)
    third = result.shows[2]
    assert third.evict is not None
    assert third.evict.plan_id == "b"
    assert third.slots_after == ["a", "c"]
    assert result.total_loads == 3


def test_evict_tie_breaks_by_smallest_plan_id():
    # a、b 都永不再用：并列时撤下方案 ID 字典序最小者（a 在槽 2）
    result = plan_stand_changes(["b", "a", "c"], 2)
    third = result.shows[2]
    assert third.evict is not None
    assert third.evict.plan_id == "a"
    assert third.evict.slot == 2
    assert third.load.slot == 2
    assert third.slots_after == ["b", "c"]


def test_single_slot_alternating():
    result = plan_stand_changes(["x", "y", "x"], 1)
    assert result.total_loads == 3
    for show in result.shows:
        assert show.load is not None
        assert show.load.slot == 1
    assert result.shows[1].evict is not None
    assert result.shows[1].evict.plan_id == "x"
    assert result.shows[-1].slots_after == ["x"]


def test_same_plan_repeated_loads_once():
    result = plan_stand_changes(["a"] * 5, 2)
    assert result.total_loads == 1
    assert [show.load is not None for show in result.shows] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(show.slots_after == ["a", None] for show in result.shows)


def test_showcase_example():
    # README 示例：A B C A B D、2 槽，总上架 5 次
    ids = ["idA", "idB", "idC", "idA", "idB", "idD"]
    result = plan_stand_changes(ids, 2)
    assert result.total_loads == 5
    trace = [
        (
            show.load.plan_id if show.load else None,
            show.evict.plan_id if show.evict else None,
            show.slots_after,
        )
        for show in result.shows
    ]
    assert trace == [
        ("idA", None, ["idA", None]),
        ("idB", None, ["idA", "idB"]),
        ("idC", "idB", ["idA", "idC"]),
        (None, None, ["idA", "idC"]),
        ("idB", "idA", ["idB", "idC"]),
        ("idD", "idB", ["idD", "idC"]),
    ]


def test_boundary_single_show_and_max_shows():
    single = plan_stand_changes(["only"], 1)
    assert single.show_count == 1
    assert single.total_loads == 1
    assert single.shows[0].slots_after == ["only"]

    many = plan_stand_changes(["only"] * MAX_SHOWS, MAX_SLOTS)
    assert many.show_count == MAX_SHOWS
    assert many.total_loads == 1
    assert many.shows[-1].slots_after == ["only"] + [None] * (MAX_SLOTS - 1)


@pytest.mark.parametrize(
    "plan_ids,slots_count,fragment",
    [
        ([], 2, "少于下限 1"),
        (["a"] * (MAX_SHOWS + 1), 2, "超过上限 500"),
        (["a"], 0, "1~20"),
        (["a"], MAX_SLOTS + 1, "1~20"),
        (["a"], 2.5, "整数"),
        (["a"], True, "整数"),
        (["a"], "2", "整数"),
        (["a", 1], 2, "字符串"),
        (["a", None], 2, "字符串"),
        (["a", True], 2, "字符串"),
        (["a", ""], 2, "非空字符串"),
        ("abc", 2, "序列"),
    ],
)
def test_invalid_inputs_raise(plan_ids, slots_count, fragment):
    with pytest.raises(StandChangeError, match=fragment):
        plan_stand_changes(plan_ids, slots_count)
