from stockroom.cache import TTLCache
from stockroom.events import EventBus
from stockroom.utils import dedupe, prefix_sums


def test_bus_survives_broken_listener():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda k, p: 1 / 0)
    bus.subscribe(lambda k, p: seen.append(p), kind="added")
    bus.emit("added", "A")
    bus.emit("removed", "A")
    assert seen == ["A"] and bus.kinds() == ["added", "removed"]


def test_ttl_cache():
    now = [0.0]
    cache = TTLCache(10, clock=lambda: now[0])
    cache.set("k", 1)
    now[0] = 10
    assert cache.get("k") == 1
    now[0] = 10.5
    assert cache.get("k") is None and len(cache) == 0


def test_utils():
    assert prefix_sums([1, 2, 3]) == [0, 1, 3, 6]
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
