"""巡演曲目单谱架换册接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client, pages: int = 1) -> str:
    """创建并保存一个方案，返回其 ID。"""
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1] * pages,
            "silence_seconds": [5] * pages,
            "page_height": 1,
            "min_turn_time": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_stand_changes_showcase_flow(client):
    """README 示例流程：4 册、2 槽、6 场，总上架 5 次。"""
    created = [_create_plan(client) for _ in range(4)]
    # 按字典序命名为 idA..idD，使并列裁决结果与创建顺序无关
    id_a, id_b, id_c, id_d = sorted(created)
    ids = [id_a, id_b, id_c, id_a, id_b, id_d]
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": ids, "stand_slots": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stand_slots"] == 2
    assert body["show_count"] == 6
    assert body["total_loads"] == 5
    assert body["request"] == {"plan_ids": ids, "stand_slots": 2}

    shows = body["shows"]
    assert [s["show_index"] for s in shows] == [1, 2, 3, 4, 5, 6]
    assert [s["plan_id"] for s in shows] == ids
    expected = [
        ({"slot": 1, "plan_id": id_a}, None, [id_a, None]),
        ({"slot": 2, "plan_id": id_b}, None, [id_a, id_b]),
        (
            {"slot": 2, "plan_id": id_c},
            {"slot": 2, "plan_id": id_b},
            [id_a, id_c],
        ),
        (None, None, [id_a, id_c]),
        (
            {"slot": 1, "plan_id": id_b},
            {"slot": 1, "plan_id": id_a},
            [id_b, id_c],
        ),
        (
            {"slot": 1, "plan_id": id_d},
            {"slot": 1, "plan_id": id_b},
            [id_d, id_c],
        ),
    ]
    for show, (load, evict, slots_after) in zip(shows, expected):
        assert show["load"] == load
        assert show["evict"] == evict
        assert show["slots_after"] == slots_after


def test_stand_changes_duplicate_ids_load_once(client):
    plan_id = _create_plan(client)
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [plan_id, plan_id, plan_id], "stand_slots": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_loads"] == 1
    assert [s["load"] is not None for s in body["shows"]] == [True, False, False]
    assert all(
        s["slots_after"] == [plan_id, None] for s in body["shows"]
    )


def test_stand_changes_eviction_tie_break_uses_plan_id(client):
    created = [_create_plan(client) for _ in range(3)]
    # 三册都只演一场：第 3 场架满时 a、b 都永不再用，撤字典序最小者
    id_a, id_b, id_c = sorted(created)
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [id_b, id_a, id_c], "stand_slots": 2},
    )
    assert resp.status_code == 200, resp.text
    shows = resp.json()["shows"]
    assert shows[2]["evict"] == {"slot": 2, "plan_id": id_a}
    assert shows[2]["load"] == {"slot": 2, "plan_id": id_c}
    assert shows[2]["slots_after"] == [id_b, id_c]


def test_stand_changes_missing_plan_returns_404(client):
    existing = _create_plan(client)
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [existing, "no-such-plan", existing], "stand_slots": 2},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "不存在" in detail
    assert "no-such-plan" in detail


def test_stand_changes_all_missing_ids_listed_once(client):
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": ["aaa", "bbb", "aaa"], "stand_slots": 2},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "aaa" in detail
    assert "bbb" in detail
    # 重复出现的不存在 ID 只列出一次
    assert detail.count("aaa") == 1


def test_stand_changes_single_show_and_slot_boundaries(client):
    plan_id = _create_plan(client)
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [plan_id], "stand_slots": 1},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["show_count"] == 1
    assert body["total_loads"] == 1
    assert body["shows"][0]["slots_after"] == [plan_id]

    # 500 场上限与 20 槽上限均合法
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [plan_id] * 500, "stand_slots": 20},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["show_count"] == 500
    assert body["total_loads"] == 1
    assert body["shows"][-1]["slots_after"] == [plan_id] + [None] * 19


@pytest.mark.parametrize(
    "payload,fragment",
    [
        # 场次数越界
        ({"plan_ids": [], "stand_slots": 2}, "1~500"),
        ({"plan_ids": ["a"] * 501, "stand_slots": 2}, "1~500"),
        # 槽位数越界（请求层即在存在性检查之前拦截）
        ({"plan_ids": ["a"], "stand_slots": 0}, "1~20"),
        ({"plan_ids": ["a"], "stand_slots": 21}, "1~20"),
        # 槽位数非整数（pydantic before 校验器拒绝）
        ({"plan_ids": ["a"], "stand_slots": 2.5}, "整数"),
        ({"plan_ids": ["a"], "stand_slots": True}, "整数"),
        ({"plan_ids": ["a"], "stand_slots": "2"}, "整数"),
        # ID 非字符串
        ({"plan_ids": [1], "stand_slots": 2}, "字符串"),
        ({"plan_ids": [None], "stand_slots": 2}, "字符串"),
        ({"plan_ids": [True], "stand_slots": 2}, "字符串"),
        ({"plan_ids": ["a", 2], "stand_slots": 2}, "字符串"),
    ],
)
def test_stand_changes_invalid_payloads_return_422(client, payload, fragment):
    resp = client.post("/api/v1/tour/stand-changes", json=payload)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


def test_stand_changes_missing_and_extra_fields_rejected(client):
    missing = client.post(
        "/api/v1/tour/stand-changes", json={"plan_ids": ["a"]}
    )
    assert missing.status_code == 422
    extra = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": ["a"], "stand_slots": 2, "note": "x"},
    )
    assert extra.status_code == 422
    no_ids = client.post("/api/v1/tour/stand-changes", json={"stand_slots": 2})
    assert no_ids.status_code == 422


def test_stand_changes_request_echoed_back(client):
    plan_id = _create_plan(client)
    payload = {"plan_ids": [plan_id, plan_id], "stand_slots": 3}
    resp = client.post("/api/v1/tour/stand-changes", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["request"] == payload


def test_stand_changes_does_not_create_or_modify_plans(client):
    ids = [_create_plan(client) for _ in range(2)]
    snapshots = {pid: client.get(f"/api/v1/plans/{pid}").json() for pid in ids}
    resp = client.post(
        "/api/v1/tour/stand-changes",
        json={"plan_ids": [ids[0], ids[1], ids[0]], "stand_slots": 1},
    )
    assert resp.status_code == 200, resp.text
    # 换册结果中出现的 ID 全部是输入 ID，没有新增方案
    returned = {
        s["load"]["plan_id"] for s in resp.json()["shows"] if s["load"]
    }
    assert returned == set(ids)
    # 原方案仍可查询且内容不变
    for pid in ids:
        assert client.get(f"/api/v1/plans/{pid}").json() == snapshots[pid]
