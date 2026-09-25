"""巡演曲目单的谱架换册核心。

背景
====
巡演期间一台谱架要在多场演出间共享若干**已保存方案**印制的乐谱册。曲目单
按**演出顺序**给出每场所需的方案 ID（允许重复：同一册可在多场使用），
谱架有 1~20 个槽位。开场时所有槽位为空；每册占一个槽位；每场所需的册
必须在该场开演前已经上架。

换册规则
========
逐场处理，每场的所需册不在架上时必须上架一册（计一次上架）：

- 有空槽：**上架到编号最小的空槽**；
- 架满：先**撤下"未来最晚再次使用"的册**（永不再用视为最晚；并列时取
  方案 ID 字典序最小者），新册上架到刚腾出的槽位。

优化目标
========
**总上架次数最少**。曲目单完全已知时，"撤下未来最晚再次使用的册"正是
Belady 最优置换策略：每次缺册都撤下最久之后才会再用的册，上架次数达到
理论最小。空槽选择与并列裁决不影响上架次数，仅保证结果确定、可复现。

求解
====
场次 n ≤ 500、槽位 s ≤ 20，一趟模拟即可：

1. 自右向左预处理 ``next_use[i]``：第 i 场所需方案下一次出现的场次序号
   （不再出现为 ``None``）；
2. 顺序模拟每场：所需册已在架上则跳过；否则有空槽取最小编号空槽，架满
   则按"最晚再次使用 → 方案 ID 字典序最小"选出被撤册，新册入其槽；
3. 记录每场的上架/撤下动作与场后各槽位内容，累计上架次数。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: 一次换册规划的场次数范围
MIN_SHOWS = 1
MAX_SHOWS = 500

#: 谱架槽位数范围
MIN_SLOTS = 1
MAX_SLOTS = 20


class StandChangeError(ValueError):
    """谱架换册输入不合法：场次数不在 1~500、槽位数不在 1~20、
    方案 ID 非字符串/为空等。"""


@dataclass(frozen=True)
class StandAction:
    """一次上架或撤下动作。"""

    slot: int  # 槽位编号（1 基）
    plan_id: str  # 被上架/撤下的方案 ID


@dataclass(frozen=True)
class ShowChange:
    """一场演出的换册记录。"""

    show_index: int  # 场次序号（1 基，按演出顺序）
    plan_id: str  # 本场所需方案 ID
    load: Optional[StandAction]  # 上架动作；所需册已在架上为 None
    evict: Optional[StandAction]  # 撤下动作；未撤册为 None
    slots_after: List[Optional[str]]  # 场后各槽位的方案 ID（空槽为 None）


@dataclass(frozen=True)
class StandChanges:
    """谱架换册结果。"""

    stand_slots: int  # 谱架槽位数
    show_count: int  # 场次数
    total_loads: int  # 总上架次数（全局最少）
    shows: List[ShowChange]  # 各场换册记录（按演出顺序）


def _validate(
    plan_ids: Sequence[str], stand_slots: int
) -> Tuple[List[str], int]:
    """校验全部输入，返回 (按演出顺序的方案 ID 列表, 槽位数)。

    :raises StandChangeError: 任一约束不满足
    """
    if not isinstance(stand_slots, int) or isinstance(stand_slots, bool):
        raise StandChangeError(f"谱架槽位数必须为整数（收到 {stand_slots!r}）")
    if not MIN_SLOTS <= stand_slots <= MAX_SLOTS:
        raise StandChangeError(
            f"谱架槽位数（{stand_slots}）必须在 {MIN_SLOTS}~{MAX_SLOTS} 之间"
        )

    if not isinstance(plan_ids, Sequence) or isinstance(plan_ids, (str, bytes)):
        raise StandChangeError("plan_ids 必须为方案 ID 序列")
    if len(plan_ids) < MIN_SHOWS:
        raise StandChangeError(
            f"场次数（{len(plan_ids)}）少于下限 {MIN_SHOWS}："
            "曲目单至少要包含一场演出"
        )
    if len(plan_ids) > MAX_SHOWS:
        raise StandChangeError(
            f"场次数（{len(plan_ids)}）超过上限 {MAX_SHOWS}："
            f"一次换册规划最多 {MAX_SHOWS} 场"
        )

    normalized: List[str] = []
    for idx, plan_id in enumerate(plan_ids):
        if not isinstance(plan_id, str) or not plan_id:
            raise StandChangeError(
                f"第 {idx + 1} 场的方案 ID 必须为非空字符串（收到 {plan_id!r}）"
            )
        normalized.append(plan_id)
    return normalized, stand_slots


def _pick_victim_slot(
    slots: Sequence[Optional[str]],
    last_seen: Dict[str, int],
    next_use: Sequence[Optional[int]],
    show_count: int,
) -> int:
    """架满时选出被撤册所在的槽位下标（0 基）。

    撤下"未来最晚再次使用"的册：永不再用视为最晚（以场次数 show_count 为
    哨兵，比任何真实场次序号都晚）；最晚并列时取方案 ID 字典序最小者。
    """
    victim = 0
    victim_distance = -1
    for idx, plan_id in enumerate(slots):
        assert plan_id is not None, "架满时不存在空槽"
        nu = next_use[last_seen[plan_id]]
        distance = nu if nu is not None else show_count
        current = slots[victim]
        assert current is not None
        if distance > victim_distance or (
            distance == victim_distance and plan_id < current
        ):
            victim = idx
            victim_distance = distance
    return victim


def plan_stand_changes(
    plan_ids: Sequence[str], stand_slots: int
) -> StandChanges:
    """为巡演曲目单规划谱架换册。

    :param plan_ids: 按演出顺序排列的已保存方案 ID（允许重复，1~500 项）；
        每项对应一场演出所需的乐谱册
    :param stand_slots: 谱架槽位数（1~20），开场时全部为空
    :return: 总上架次数最少的换册结果（含每场的上架/撤下动作与场后各槽位
        方案 ID）
    :raises StandChangeError: 场次数不在 1~500、槽位数不在 1~20、
        方案 ID 非字符串或为空
    """
    ids, slots_count = _validate(plan_ids, stand_slots)
    n = len(ids)

    # next_use[i]：第 i 场所需方案下一次出现的场次序号（0 基），不再出现为 None
    next_use: List[Optional[int]] = [None] * n
    last_occurrence: Dict[str, int] = {}
    for i in range(n - 1, -1, -1):
        pid = ids[i]
        next_use[i] = last_occurrence.get(pid)
        last_occurrence[pid] = i

    slots: List[Optional[str]] = [None] * slots_count
    on_stand: Dict[str, int] = {}  # 方案 ID -> 槽位下标（0 基）
    last_seen: Dict[str, int] = {}  # 方案 ID -> 最近一次所需场次序号（0 基）
    total_loads = 0
    shows: List[ShowChange] = []

    for i, pid in enumerate(ids):
        last_seen[pid] = i
        load_action: Optional[StandAction] = None
        evict_action: Optional[StandAction] = None
        if pid not in on_stand:
            if None in slots:
                slot_idx = slots.index(None)  # 编号最小的空槽
            else:
                slot_idx = _pick_victim_slot(slots, last_seen, next_use, n)
                evicted = slots[slot_idx]
                assert evicted is not None
                del on_stand[evicted]
                evict_action = StandAction(slot=slot_idx + 1, plan_id=evicted)
            slots[slot_idx] = pid
            on_stand[pid] = slot_idx
            total_loads += 1
            load_action = StandAction(slot=slot_idx + 1, plan_id=pid)
        shows.append(
            ShowChange(
                show_index=i + 1,
                plan_id=pid,
                load=load_action,
                evict=evict_action,
                slots_after=list(slots),
            )
        )

    return StandChanges(
        stand_slots=slots_count,
        show_count=n,
        total_loads=total_loads,
        shows=shows,
    )
