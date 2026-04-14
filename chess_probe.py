"""Minimal legality-probe experiment for chess positions.

This is a dependency-free prototype of the project proposal. It does not load a
Chess-GPT checkpoint; instead it builds a controlled probe task with chess-board
features that stand in for progressively richer residual-stream representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
from typing import Iterable


WHITE = "w"
BLACK = "b"
EMPTY = "."
PIECES = "PNBRQKpnbrqk"
PIECE_VALUES = {
    "P": 1.0,
    "N": 3.0,
    "B": 3.0,
    "R": 5.0,
    "Q": 9.0,
    "K": 0.0,
    "p": -1.0,
    "n": -3.0,
    "b": -3.0,
    "r": -5.0,
    "q": -9.0,
    "k": 0.0,
}


@dataclass(frozen=True)
class Move:
    start: int
    end: int
    promotion: str | None = None

    def uci(self) -> str:
        text = square_name(self.start) + square_name(self.end)
        return text + (self.promotion.lower() if self.promotion else "")


@dataclass
class Board:
    squares: list[str]
    turn: str = WHITE

    @classmethod
    def initial(cls) -> "Board":
        rows = [
            "rnbqkbnr",
            "pppppppp",
            "........",
            "........",
            "........",
            "........",
            "PPPPPPPP",
            "RNBQKBNR",
        ]
        return cls([piece for row in rows for piece in row], WHITE)

    def copy(self) -> "Board":
        return Board(self.squares[:], self.turn)

    def apply(self, move: Move) -> "Board":
        board = self.copy()
        piece = board.squares[move.start]
        board.squares[move.start] = EMPTY
        board.squares[move.end] = move.promotion or piece
        board.turn = other(self.turn)
        return board

    def fenish(self) -> str:
        rows = []
        for rank in range(8):
            row = "".join(self.squares[rank * 8 : rank * 8 + 8])
            rows.append(row)
        return "/".join(rows) + " " + self.turn


def other(color: str) -> str:
    return BLACK if color == WHITE else WHITE


def is_white(piece: str) -> bool:
    return piece.isupper()


def piece_color(piece: str) -> str | None:
    if piece == EMPTY:
        return None
    return WHITE if is_white(piece) else BLACK


def rank_of(square: int) -> int:
    return square // 8


def file_of(square: int) -> int:
    return square % 8


def to_square(rank: int, file: int) -> int | None:
    if 0 <= rank < 8 and 0 <= file < 8:
        return rank * 8 + file
    return None


def square_name(square: int) -> str:
    return "abcdefgh"[file_of(square)] + str(8 - rank_of(square))


def pseudo_moves(board: Board) -> list[Move]:
    moves: list[Move] = []
    for square, piece in enumerate(board.squares):
        if piece == EMPTY or piece_color(piece) != board.turn:
            continue
        kind = piece.upper()
        if kind == "P":
            moves.extend(pawn_moves(board, square, piece))
        elif kind == "N":
            moves.extend(jump_moves(board, square, piece, KNIGHT_STEPS))
        elif kind == "B":
            moves.extend(sliding_moves(board, square, piece, BISHOP_DIRS))
        elif kind == "R":
            moves.extend(sliding_moves(board, square, piece, ROOK_DIRS))
        elif kind == "Q":
            moves.extend(sliding_moves(board, square, piece, BISHOP_DIRS + ROOK_DIRS))
        elif kind == "K":
            moves.extend(jump_moves(board, square, piece, KING_STEPS))
    return moves


def legal_moves(board: Board) -> list[Move]:
    legal: list[Move] = []
    for move in pseudo_moves(board):
        next_board = board.apply(move)
        if not is_in_check(next_board, board.turn):
            legal.append(move)
    return legal


def pawn_moves(board: Board, square: int, piece: str) -> list[Move]:
    color = piece_color(piece)
    direction = -1 if color == WHITE else 1
    start_rank = 6 if color == WHITE else 1
    promotion_rank = 0 if color == WHITE else 7
    rank, file = rank_of(square), file_of(square)
    moves: list[Move] = []

    one = to_square(rank + direction, file)
    if one is not None and board.squares[one] == EMPTY:
        moves.append(promote_if_needed(square, one, promotion_rank, color))
        two = to_square(rank + 2 * direction, file)
        if rank == start_rank and two is not None and board.squares[two] == EMPTY:
            moves.append(Move(square, two))

    for df in (-1, 1):
        target = to_square(rank + direction, file + df)
        if target is None:
            continue
        target_piece = board.squares[target]
        if target_piece != EMPTY and piece_color(target_piece) != color:
            moves.append(promote_if_needed(square, target, promotion_rank, color))
    return moves


def promote_if_needed(start: int, end: int, promotion_rank: int, color: str) -> Move:
    if rank_of(end) == promotion_rank:
        return Move(start, end, "Q" if color == WHITE else "q")
    return Move(start, end)


KNIGHT_STEPS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def jump_moves(board: Board, square: int, piece: str, steps: Iterable[tuple[int, int]]) -> list[Move]:
    color = piece_color(piece)
    moves: list[Move] = []
    rank, file = rank_of(square), file_of(square)
    for dr, df in steps:
        target = to_square(rank + dr, file + df)
        if target is None:
            continue
        target_piece = board.squares[target]
        if target_piece == EMPTY or piece_color(target_piece) != color:
            moves.append(Move(square, target))
    return moves


def sliding_moves(board: Board, square: int, piece: str, dirs: Iterable[tuple[int, int]]) -> list[Move]:
    color = piece_color(piece)
    moves: list[Move] = []
    rank, file = rank_of(square), file_of(square)
    for dr, df in dirs:
        next_rank, next_file = rank + dr, file + df
        while True:
            target = to_square(next_rank, next_file)
            if target is None:
                break
            target_piece = board.squares[target]
            if target_piece == EMPTY:
                moves.append(Move(square, target))
            else:
                if piece_color(target_piece) != color:
                    moves.append(Move(square, target))
                break
            next_rank += dr
            next_file += df
    return moves


def is_in_check(board: Board, color: str) -> bool:
    king = "K" if color == WHITE else "k"
    try:
        king_square = board.squares.index(king)
    except ValueError:
        return True
    return is_attacked(board, king_square, other(color))


def is_attacked(board: Board, square: int, by_color: str) -> bool:
    rank, file = rank_of(square), file_of(square)

    pawn_dir = 1 if by_color == WHITE else -1
    pawn = "P" if by_color == WHITE else "p"
    for df in (-1, 1):
        source = to_square(rank + pawn_dir, file + df)
        if source is not None and board.squares[source] == pawn:
            return True

    knight = "N" if by_color == WHITE else "n"
    for dr, df in KNIGHT_STEPS:
        source = to_square(rank + dr, file + df)
        if source is not None and board.squares[source] == knight:
            return True

    king = "K" if by_color == WHITE else "k"
    for dr, df in KING_STEPS:
        source = to_square(rank + dr, file + df)
        if source is not None and board.squares[source] == king:
            return True

    attackers = {
        (-1, -1): ("B", "Q"),
        (-1, 1): ("B", "Q"),
        (1, -1): ("B", "Q"),
        (1, 1): ("B", "Q"),
        (-1, 0): ("R", "Q"),
        (1, 0): ("R", "Q"),
        (0, -1): ("R", "Q"),
        (0, 1): ("R", "Q"),
    }
    for (dr, df), kinds in attackers.items():
        next_rank, next_file = rank + dr, file + df
        while True:
            source = to_square(next_rank, next_file)
            if source is None:
                break
            piece = board.squares[source]
            if piece != EMPTY:
                if piece_color(piece) == by_color and piece.upper() in kinds:
                    return True
                break
            next_rank += dr
            next_file += df
    return False


def random_position(rng: random.Random, max_plies: int) -> Board:
    board = Board.initial()
    for _ in range(rng.randint(0, max_plies)):
        moves = legal_moves(board)
        if not moves:
            break
        board = board.apply(rng.choice(moves))
    return board


def corrupt_move(board: Board, move: Move, rng: random.Random) -> Move:
    legal_uci = {m.uci() for m in legal_moves(board)}
    for _ in range(128):
        if rng.random() < 0.5:
            candidate = Move(move.start, rng.randrange(64))
        else:
            own = [i for i, p in enumerate(board.squares) if piece_color(p) == board.turn]
            candidate = Move(rng.choice(own), rng.randrange(64))
        if candidate.start != candidate.end and candidate.uci() not in legal_uci:
            return candidate
    return Move(move.start, (move.end + 9) % 64)


@dataclass
class Example:
    board: Board
    move: Move
    label: int


def make_dataset(seed: int, positions: int, max_plies: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    seen_positions = 0
    while seen_positions < positions:
        board = random_position(rng, max_plies)
        moves = legal_moves(board)
        if not moves:
            continue
        legal = rng.choice(moves)
        examples.append(Example(board, legal, 1))
        examples.append(Example(board, corrupt_move(board, legal, rng), 0))
        seen_positions += 1
    rng.shuffle(examples)
    return examples


def material_features(board: Board, move: Move) -> list[float]:
    del move
    total = sum(PIECE_VALUES[p] for p in board.squares if p != EMPTY)
    own_material = total if board.turn == WHITE else -total
    return [1.0, own_material / 39.0]


def board_features(board: Board, move: Move) -> list[float]:
    del move
    features: list[float] = []
    for piece in PIECES:
        features.extend(1.0 if square == piece else 0.0 for square in board.squares)
    features.append(1.0 if board.turn == WHITE else -1.0)
    return features


def candidate_features(board: Board, move: Move) -> list[float]:
    piece = board.squares[move.start]
    target = board.squares[move.end]
    same_color_capture = piece != EMPTY and target != EMPTY and piece_color(piece) == piece_color(target)
    empty_source = piece == EMPTY
    rank_delta = rank_of(move.end) - rank_of(move.start)
    file_delta = file_of(move.end) - file_of(move.start)
    return [
        move.start / 63.0,
        move.end / 63.0,
        rank_delta / 7.0,
        file_delta / 7.0,
        abs(rank_delta) / 7.0,
        abs(file_delta) / 7.0,
        1.0 if empty_source else 0.0,
        1.0 if same_color_capture else 0.0,
        1.0 if target != EMPTY and not same_color_capture else 0.0,
    ]


def tactical_features(board: Board, move: Move) -> list[float]:
    piece = board.squares[move.start]
    if piece == EMPTY or piece_color(piece) != board.turn:
        return [0.0, 0.0, 1.0]
    pseudo = {m.uci() for m in pseudo_moves(board)}
    leaves_king_safe = not is_in_check(board.apply(move), board.turn) if move.uci() in pseudo else False
    return [
        1.0 if move.uci() in pseudo else 0.0,
        1.0 if leaves_king_safe else 0.0,
        0.0,
    ]


FEATURE_BLOCKS = [
    ("material", material_features),
    ("board_state", board_features),
    ("candidate_move", candidate_features),
    ("tactical_legality", tactical_features),
]


def features_for(example: Example, depth: int) -> list[float]:
    features: list[float] = []
    for _, featurizer in FEATURE_BLOCKS[:depth]:
        features.extend(featurizer(example.board, example.move))
    return features


class LogisticProbe:
    def __init__(self, n_features: int) -> None:
        self.weights = [0.0] * n_features
        self.bias = 0.0

    def probability(self, x: list[float]) -> float:
        z = self.bias + sum(weight * value for weight, value in zip(self.weights, x))
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def fit(self, xs: list[list[float]], ys: list[int], epochs: int, lr: float, l2: float) -> None:
        for _ in range(epochs):
            for x, y in zip(xs, ys):
                pred = self.probability(x)
                err = pred - y
                for i, value in enumerate(x):
                    self.weights[i] -= lr * (err * value + l2 * self.weights[i])
                self.bias -= lr * err

    def predict(self, x: list[float]) -> int:
        return 1 if self.probability(x) >= 0.5 else 0


def evaluate(model: LogisticProbe, xs: list[list[float]], ys: list[int]) -> tuple[float, float]:
    correct = 0
    loss = 0.0
    for x, y in zip(xs, ys):
        prob = min(max(model.probability(x), 1e-8), 1.0 - 1e-8)
        correct += model.predict(x) == y
        loss += -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
    return correct / len(ys), loss / len(ys)


def run(seed: int, positions: int, max_plies: int, epochs: int) -> list[dict[str, float | str | int]]:
    data = make_dataset(seed, positions, max_plies)
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]
    rows: list[dict[str, float | str | int]] = []

    for depth, (name, _) in enumerate(FEATURE_BLOCKS, start=1):
        train_x = [features_for(example, depth) for example in train]
        train_y = [example.label for example in train]
        test_x = [features_for(example, depth) for example in test]
        test_y = [example.label for example in test]
        probe = LogisticProbe(len(train_x[0]))
        probe.fit(train_x, train_y, epochs=epochs, lr=0.06, l2=0.0005)
        train_acc, train_loss = evaluate(probe, train_x, train_y)
        test_acc, test_loss = evaluate(probe, test_x, test_y)
        rows.append(
            {
                "block": name,
                "features": len(train_x[0]),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "train_loss": train_loss,
            }
        )
    return rows


def print_results(rows: list[dict[str, float | str | int]]) -> None:
    print("Legality probe results")
    print("block              features  train_acc  test_acc  test_loss")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['block']:<18}"
            f"{row['features']:>8}  "
            f"{row['train_accuracy']:.3f}      "
            f"{row['test_accuracy']:.3f}     "
            f"{row['test_loss']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal chess legality probe.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--positions", type=int, default=450)
    parser.add_argument("--max-plies", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    rows = run(args.seed, args.positions, args.max_plies, args.epochs)
    print_results(rows)


if __name__ == "__main__":
    main()
