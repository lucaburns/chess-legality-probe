"""Smoke tests for the minimal chess legality-probe project."""

from chess_probe import Board, Move, legal_moves, make_dataset, run


def test_initial_position_has_twenty_legal_moves() -> None:
    assert len(legal_moves(Board.initial())) == 20


def test_basic_uci_formatting() -> None:
    assert Move(52, 36).uci() == "e2e4"


def test_dataset_is_balanced() -> None:
    data = make_dataset(seed=1, positions=12, max_plies=12)
    assert len(data) == 24
    assert sum(example.label for example in data) == 12


def test_probe_runs_end_to_end() -> None:
    rows = run(seed=2, positions=40, max_plies=10, epochs=2)
    assert len(rows) == 4
    assert all(0.0 <= row["test_accuracy"] <= 1.0 for row in rows)


if __name__ == "__main__":
    test_initial_position_has_twenty_legal_moves()
    test_basic_uci_formatting()
    test_dataset_is_balanced()
    test_probe_runs_end_to_end()
    print("All smoke tests passed.")
