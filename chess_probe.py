"""Minimal legality-probe experiment for chess positions.

This is a dependency-free prototype of the project proposal. It does not load a
Chess-GPT checkpoint; instead it builds a controlled probe task with chess-board
features that stand in for progressively richer residual-stream representations.

Changes from the first draft:
- Illegal-move sampling prefers "hard negatives" -- pseudo-legal moves that
  leave the king in check -- so the task isn't trivially solvable from the
  candidate-square features alone.
- Feature ablations are reported both cumulatively (what you can predict given
  everything up to depth k) and individually (what each block contributes on
  its own), so the ladder isn't dominated by the tactical-legality oracle.
- A majority-class baseline is printed alongside the probes.
- K-fold cross-validation gives a noise estimate instead of a single split.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
from typing import Callable, Iterable


WHITE = "w"
BLACK = "b"
EMPTY = "."
PIECES = "PNBRQKpnbrqk"
PIECE_VALUES = {
    "P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0, "K": 0.0,
    "p": -1.0, "n": -3.0, "b": -3.0, "r": -5.0, "q": -9.0, "k": 0.0,
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


def sample_illegal_move(board: Board, rng: random.Random) -> Move | None:
    """Prefer hard negatives: pseudo-legal moves that leave own king in check.

    Falls back to off-board / same-color-capture style garbage only if no
    pseudo-legal-but-illegal candidates exist (e.g. rare forced positions).
    """
    pseudo = pseudo_moves(board)
    legal_uci = {m.uci() for m in legal_moves(board)}
    hard = [m for m in pseudo if m.uci() not in legal_uci]
    if hard:
        return rng.choice(hard)

    own = [i for i, p in enumerate(board.squares) if piece_color(p) == board.turn]
    if not own:
        return None
    for _ in range(128):
        start = rng.choice(own)
        end = rng.randrange(64)
        if start == end:
            continue
        candidate = Move(start, end)
        if candidate.uci() not in legal_uci:
            return candidate
    return None


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
        illegal = sample_illegal_move(board, rng)
        if illegal is None:
            continue
        legal = rng.choice(moves)
        examples.append(Example(board, legal, 1))
        examples.append(Example(board, illegal, 0))
        seen_positions += 1
    rng.shuffle(examples)
    return examples


def material_features(board: Board, move: Move) -> list[float]:
    del move
    total = sum(PIECE_VALUES[p] for p in board.squares if p != EMPTY)
    own_material = total if board.turn == WHITE else -total
    return [1.0, own_material / 39.0]


def candidate_features(board: Board, move: Move) -> list[float]:
    """Move-geometry features. Always included: predicting legality of move X
    without seeing X at all is a degenerate task."""
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


def piece_context_features(board: Board, move: Move) -> list[float]:
    """What's at the source, destination, and immediate neighbourhood of the move.

    Avoids the full 769-dim board-state blowup (which a tiny probe can't fit
    on a few hundred examples) while still giving the probe enough context to
    resolve many pseudo-legal-but-illegal cases.
    """
    features: list[float] = []
    for square in (move.start, move.end):
        piece = board.squares[square]
        for letter in PIECES:
            features.append(1.0 if piece == letter else 0.0)
        features.append(1.0 if piece == EMPTY else 0.0)
    # Count same-colour pieces adjacent to destination (crude defender signal).
    dr_df = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if (dr, df) != (0, 0)]
    own_adjacent = 0
    enemy_adjacent = 0
    for dr, df in dr_df:
        sq = to_square(rank_of(move.end) + dr, file_of(move.end) + df)
        if sq is None:
            continue
        p = board.squares[sq]
        if p == EMPTY:
            continue
        if piece_color(p) == board.turn:
            own_adjacent += 1
        else:
            enemy_adjacent += 1
    features.append(own_adjacent / 8.0)
    features.append(enemy_adjacent / 8.0)
    return features


def king_safety_features(board: Board, move: Move) -> list[float]:
    """King-safety signal without directly leaking the legal/pseudo-legal label.

    We report (a) whether the moving piece is pinned along a rank/file/diagonal
    to our king (a proxy for 'this move likely leaves the king in check') and
    (b) how many enemy pieces currently attack our king's square. These are
    heuristic features that a strong representation might encode, not the
    ground-truth legality oracle.
    """
    color = board.turn
    king = "K" if color == WHITE else "k"
    try:
        king_sq = board.squares.index(king)
    except ValueError:
        return [0.0, 0.0, 0.0]

    piece = board.squares[move.start]
    on_king_ray = 0.0
    if piece != EMPTY and piece_color(piece) == color:
        kr, kf = rank_of(king_sq), file_of(king_sq)
        sr, sf = rank_of(move.start), file_of(move.start)
        if kr == sr or kf == sf or abs(kr - sr) == abs(kf - sf):
            on_king_ray = 1.0

    attackers_on_king = 0
    for sq in range(64):
        p = board.squares[sq]
        if p != EMPTY and piece_color(p) == other(color):
            if _single_piece_attacks(board, sq, king_sq):
                attackers_on_king += 1

    return [on_king_ray, attackers_on_king / 4.0, 1.0 if board.turn == WHITE else 0.0]


def _single_piece_attacks(board: Board, source: int, target: int) -> bool:
    piece = board.squares[source]
    if piece == EMPTY:
        return False
    kind = piece.upper()
    sr, sf = rank_of(source), file_of(source)
    tr, tf = rank_of(target), file_of(target)
    dr, df = tr - sr, tf - sf
    if kind == "P":
        direction = -1 if is_white(piece) else 1
        return dr == direction and abs(df) == 1
    if kind == "N":
        return (abs(dr), abs(df)) in {(1, 2), (2, 1)}
    if kind == "K":
        return max(abs(dr), abs(df)) == 1
    if kind in {"B", "R", "Q"}:
        if kind == "B" and abs(dr) != abs(df):
            return False
        if kind == "R" and dr != 0 and df != 0:
            return False
        if kind == "Q" and not (dr == 0 or df == 0 or abs(dr) == abs(df)):
            return False
        step_r = (dr > 0) - (dr < 0)
        step_f = (df > 0) - (df < 0)
        r, f = sr + step_r, sf + step_f
        while (r, f) != (tr, tf):
            sq = to_square(r, f)
            if sq is None or board.squares[sq] != EMPTY:
                return False
            r += step_r
            f += step_f
        return True
    return False


FEATURE_BLOCKS: list[tuple[str, Callable[[Board, Move], list[float]]]] = [
    ("material", material_features),
    ("candidate_move", candidate_features),
    ("piece_context", piece_context_features),
    ("king_safety", king_safety_features),
]


def features_cumulative(example: Example, depth: int) -> list[float]:
    features: list[float] = []
    for _, featurizer in FEATURE_BLOCKS[:depth]:
        features.extend(featurizer(example.board, example.move))
    return features


def features_single(example: Example, index: int) -> list[float]:
    _, featurizer = FEATURE_BLOCKS[index]
    return featurizer(example.board, example.move)


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


def majority_baseline(ys: list[int]) -> float:
    if not ys:
        return 0.0
    ones = sum(ys)
    return max(ones, len(ys) - ones) / len(ys)


def kfold_indices(n: int, k: int, rng: random.Random) -> list[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(order):
        folds[i % k].append(idx)
    return folds


def fit_and_score(
    data: list[Example],
    featurizer: Callable[[Example], list[float]],
    folds: list[list[int]],
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[float, float, int]:
    xs_all = [featurizer(example) for example in data]
    ys_all = [example.label for example in data]
    dim = len(xs_all[0])
    # High-dim blocks need a smaller per-feature step (SGD gradient norm grows
    # with sqrt(dim) under standardised features) and stronger regularisation.
    scaled_lr = lr / math.sqrt(max(dim, 1))
    scaled_l2 = l2 * max(1.0, dim / 16.0)
    accs: list[float] = []
    for i, test_idx in enumerate(folds):
        train_idx = [j for f, fold in enumerate(folds) if f != i for j in fold]
        train_x = [xs_all[j] for j in train_idx]
        train_y = [ys_all[j] for j in train_idx]
        test_x = [xs_all[j] for j in test_idx]
        test_y = [ys_all[j] for j in test_idx]

        means = [sum(col) / len(train_x) for col in zip(*train_x)]
        stds = [
            math.sqrt(sum((row[k] - means[k]) ** 2 for row in train_x) / len(train_x)) or 1.0
            for k in range(dim)
        ]
        train_x = [[(row[k] - means[k]) / stds[k] for k in range(dim)] for row in train_x]
        test_x = [[(row[k] - means[k]) / stds[k] for k in range(dim)] for row in test_x]

        probe = LogisticProbe(dim)
        probe.fit(train_x, train_y, epochs=epochs, lr=scaled_lr, l2=scaled_l2)
        acc, _ = evaluate(probe, test_x, test_y)
        accs.append(acc)
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)
    return mean, math.sqrt(var), dim


def run(seed: int, positions: int, max_plies: int, epochs: int, folds_k: int) -> None:
    data = make_dataset(seed, positions, max_plies)
    fold_rng = random.Random(seed + 1)
    folds = kfold_indices(len(data), folds_k, fold_rng)
    baseline = majority_baseline([e.label for e in data])

    print(f"Dataset: {len(data)} examples ({sum(e.label for e in data)} legal, "
          f"{len(data) - sum(e.label for e in data)} illegal).")
    print(f"Majority-class baseline: {baseline:.3f}")
    print()

    print("Cumulative feature ladder (features up to and including depth)")
    print("depth  block              features  test_acc  ± std")
    print("-" * 56)
    for depth, (name, _) in enumerate(FEATURE_BLOCKS, start=1):
        mean, std, dim = fit_and_score(
            data,
            lambda ex, d=depth: features_cumulative(ex, d),
            folds,
            epochs=epochs,
            lr=0.02,
            l2=0.001,
        )
        print(f"{depth:>5}  {name:<18}{dim:>8}  {mean:.3f}     ±{std:.3f}")

    print()
    print("Individual block contribution (that block's features only)")
    print("block              features  test_acc  ± std")
    print("-" * 49)
    for idx, (name, _) in enumerate(FEATURE_BLOCKS):
        mean, std, dim = fit_and_score(
            data,
            lambda ex, i=idx: features_single(ex, i),
            folds,
            epochs=epochs,
            lr=0.02,
            l2=0.001,
        )
        print(f"{name:<18}{dim:>8}  {mean:.3f}     ±{std:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal chess legality probe.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--positions", type=int, default=450)
    parser.add_argument("--max-plies", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run(args.seed, args.positions, args.max_plies, args.epochs, args.folds)


if __name__ == "__main__":
    main()
