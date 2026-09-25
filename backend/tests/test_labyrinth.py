"""The dashboard's labyrinth: rooms on a grid, the passages carved between them, today's
thread, and the Minotaur's room."""

from collections import deque
from datetime import date, timedelta

import pytest

from app.scheduling.labyrinth import (
    Maze,
    Room,
    carve,
    lair,
    route,
    thread,
    topic_rooms,
    visited,
)
from app.scheduling.mastery import Standing

DAY = date(2026, 9, 25)


def reachable(maze: Maze) -> set[int]:
    seen = {maze.entrance}
    queue = deque([maze.entrance])
    while queue:
        cell = queue.popleft()
        for following in maze.neighbours(cell):
            if following not in seen and maze.joined(cell, following):
                seen.add(following)
                queue.append(following)
    return seen


def test_the_same_topics_always_give_the_same_map() -> None:
    assert carve(17) == carve(17)
    assert carve(17) != carve(17, seed=1)


@pytest.mark.parametrize("rooms", range(1, 40))
def test_every_room_can_be_reached_and_by_exactly_one_way(rooms: int) -> None:
    maze = carve(rooms)

    assert maze is not None
    assert reachable(maze) == set(range(rooms))
    # Connected, with one passage fewer than rooms: a tree, so no loops
    assert len(maze.passages) == rooms - 1


@pytest.mark.parametrize(
    ("rooms", "rows", "entrance"), [(1, 1, 0), (6, 1, 0), (7, 2, 6), (17, 3, 12), (18, 3, 12)]
)
def test_rooms_fill_a_grid_six_wide_with_the_entrance_at_the_bottom_left(
    rooms: int, rows: int, entrance: int
) -> None:
    maze = carve(rooms)

    assert maze is not None
    assert (maze.columns, maze.rows, maze.entrance) == (6, rows, entrance)


def test_passages_join_neighbouring_rooms_only() -> None:
    maze = carve(17)

    assert maze is not None
    for a, b in maze.passages:
        assert a < b < 17
        assert (b - a == 1 and a // 6 == b // 6) or b - a == 6


def test_there_is_no_labyrinth_without_rooms() -> None:
    assert carve(0) is None


def test_a_route_is_the_one_way_between_two_rooms() -> None:
    maze = carve(17)
    assert maze is not None

    path = route(maze, maze.entrance, 5)

    assert (path[0], path[-1]) == (maze.entrance, 5)
    assert all(maze.joined(a, b) for a, b in zip(path, path[1:], strict=False))
    assert route(maze, 5, maze.entrance) == path[::-1]
    assert route(maze, 5, 5) == [5]


def test_the_thread_runs_from_the_entrance_through_each_room_visited() -> None:
    maze = carve(17)
    assert maze is not None

    path = thread(maze, [3, 9, 3])

    assert path[0] == maze.entrance
    assert all(maze.joined(a, b) for a, b in zip(path, path[1:], strict=False))
    # The visits come in order along it
    rest = iter(path)
    assert all(visit in rest for visit in [3, 9, 3])
    assert path[-1] == 3
    assert thread(maze, []) == []


def standing(
    question_id: int, topic_id: int | None, score: float | None = None, due: date | None = None
) -> Standing:
    state = {"card_id": question_id} if score is not None else None
    return Standing(question_id, topic_id, 3, score, state, due, 1.0 if score else 0.0)


def test_each_topic_with_questions_in_the_library_is_a_room() -> None:
    rooms = topic_rooms(
        [
            standing(1, 7, score=0.8, due=DAY),
            standing(2, 7),
            standing(3, 3, score=0.5, due=DAY + timedelta(days=2)),
            standing(4, None, score=1.0, due=DAY),
        ],
        DAY,
    )

    # In topic order; the question without a topic has no room
    assert rooms == [
        Room(topic_id=3, questions=1, practised=1, mastery=0.5, due=0),
        Room(topic_id=7, questions=2, practised=1, mastery=0.4, due=1),
    ]


def test_the_minotaur_waits_in_the_weakest_room() -> None:
    rooms = [
        Room(topic_id=1, questions=2, practised=1, mastery=0.3, due=0),
        Room(topic_id=2, questions=1, practised=0, mastery=0.0, due=0),
        Room(topic_id=3, questions=1, practised=1, mastery=0.0, due=1),
        Room(topic_id=4, questions=1, practised=0, mastery=0.0, due=0),
    ]

    # The lowest mastery, then the room practised least, then the first topic
    assert lair(rooms) == 1
    assert lair([]) is None


def test_today_s_visits_follow_the_answers_room_by_room() -> None:
    rooms = [
        Room(topic_id=3, questions=1, practised=1, mastery=0.5, due=0),
        Room(topic_id=7, questions=2, practised=1, mastery=0.4, due=0),
    ]

    # Two answers running in one room count once; a question without a room is passed over
    assert visited([7, 7, None, 3, 99, 7], rooms) == [1, 0, 1]
