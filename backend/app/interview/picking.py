"""Which library questions an interview asks.

The practice picker's order decides (`app.scheduling.picker`): questions due first, then new
ones from the weakest topic, then practice ahead of the schedule. An interview asks several at
once, so it takes the picker's choice, sets that question aside and asks again. Across topics
it sets the question's whole topic aside as well, until every topic has had a question.
"""

from datetime import date

from app.scheduling.mastery import Standing
from app.scheduling.picker import choose


def plan(questions: list[Standing], day: date, size: int, topic_id: int | None) -> list[int]:
    """Up to `size` questions to ask on `day`, in the order to ask them: from one topic, or
    from as many topics as there are when `topic_id` is None."""
    pool = [question for question in questions if topic_id is None or question.topic_id == topic_id]
    topics = {question.question_id: question.topic_id for question in pool}
    picked: list[int] = []
    asked_about: set[int | None] = set()
    while len(picked) < size:
        left = [question for question in pool if question.question_id not in picked]
        unasked = [question for question in left if question.topic_id not in asked_about]
        pick = choose(unasked or left, day)
        if pick is None:
            break
        picked.append(pick.question_id)
        asked_about.add(topics[pick.question_id])
    return picked
