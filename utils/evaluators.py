from abstractions import Action, State, Q


def best_q_value(q: Q, state: State, actions: list[Action]) -> float:
    ans = float("-inf")
    for a in actions:
        if (qa := q.get((state, a), 0)) > ans:
            ans = qa

    return ans
