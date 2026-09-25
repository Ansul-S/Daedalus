"""The dashboard's labyrinth: a room for each topic, joined by passages.

Rooms fill a grid six wide, row by row in the order of their topics, and the rest of the last
row is solid stone. A depth-first search from the entrance, the room at the bottom left,
carves the passages: every room can be reached, and by exactly one way. The search is seeded,
so the same topics always give the same map.

The Minotaur waits in the weakest room, judged as the picker judges topics: the lowest
mastery, then the one practised least.
"""

import random
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from app.scheduling.mastery import Standing, topic_mastery

COLUMNS = 6
# Attention Is All You Need is arXiv 1706.03762
SEED = 1706


@dataclass(frozen=True)
class Maze:
    columns: int
    rows: int
    # Cells are numbered row by row from the top left; the first `rooms` of them are rooms
    rooms: int
    # Neighbouring cells joined by a passage, the lower-numbered first
    passages: frozenset[tuple[int, int]]
    entrance: int

    def neighbours(self, cell: int) -> list[int]:
        """The rooms next to a cell: left, right, up and down."""
        x, y = cell % self.columns, cell // self.columns
        cells = [
            cell - 1 if x > 0 else None,
            cell + 1 if x < self.columns - 1 else None,
            cell - self.columns if y > 0 else None,
            cell + self.columns,
        ]
        return [n for n in cells if n is not None and n < self.rooms]

    def joined(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.passages


def carve(rooms: int, columns: int = COLUMNS, seed: int = SEED) -> Maze | None:
    """A maze through `rooms` rooms, or None when there are none."""
    if rooms <= 0:
        return None
    rows = -(-rooms // columns)
    entrance = (rows - 1) * columns
    maze = Maze(columns, rows, rooms, frozenset(), entrance)
    choose = random.Random(seed)
    passages: set[tuple[int, int]] = set()
    seen = {entrance}
    stack = [entrance]
    while stack:
        cell = stack[-1]
        fresh = [n for n in maze.neighbours(cell) if n not in seen]
        if not fresh:
            stack.pop()
            continue
        # random() rather than choice(): its sequence for a seed never changes between versions
        following = fresh[int(choose.random() * len(fresh))]
        passages.add((min(cell, following), max(cell, following)))
        seen.add(following)
        stack.append(following)
    return Maze(columns, rows, rooms, frozenset(passages), entrance)


def route(maze: Maze, start: int, end: int) -> list[int]:
    """The cells from one room to another along the passages, both ends included."""
    previous = {start: start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == end:
            break
        for following in maze.neighbours(cell):
            if following not in previous and maze.joined(cell, following):
                previous[following] = cell
                queue.append(following)
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def thread(maze: Maze, visits: Sequence[int]) -> list[int]:
    """Ariadne's thread: from the entrance through each room visited, in turn. Empty when
    there were no visits."""
    if not visits:
        return []
    path = [maze.entrance]
    for visit in visits:
        path += route(maze, path[-1], visit)[1:]
    return path


@dataclass(frozen=True)
class Room:
    topic_id: int
    questions: int
    practised: int
    mastery: float
    due: int


def topic_rooms(questions: Iterable[Standing], day: date) -> list[Room]:
    """A room for each topic with questions in the library, in topic order. Questions
    without a topic have no room."""
    questions = [question for question in questions if question.topic_id is not None]
    mastery = topic_mastery(questions)
    rooms = []
    for topic_id in sorted({question.topic_id for question in questions}):
        inside = [question for question in questions if question.topic_id == topic_id]
        rooms.append(
            Room(
                topic_id=topic_id,
                questions=len(inside),
                practised=sum(question.practised for question in inside),
                mastery=mastery[topic_id],
                due=sum(question.due is not None and question.due <= day for question in inside),
            )
        )
    return rooms


def lair(rooms: Sequence[Room]) -> int | None:
    """The cell of the weakest room, or None when there are no rooms."""
    if not rooms:
        return None
    return min(
        range(len(rooms)),
        key=lambda cell: (rooms[cell].mastery, rooms[cell].practised, rooms[cell].topic_id),
    )


def visited(topics: Iterable[int | None], rooms: Sequence[Room]) -> list[int]:
    """The rooms of the questions answered, in order, a room answered in twice running
    counted once. Questions without a room are passed over."""
    cells = {room.topic_id: cell for cell, room in enumerate(rooms)}
    path: list[int] = []
    for topic_id in topics:
        cell = cells.get(topic_id) if topic_id is not None else None
        if cell is not None and (not path or path[-1] != cell):
            path.append(cell)
    return path
