"""Mini NPU Simulator: Python 3.8+, standard library only."""
import argparse
import json
import math
from pathlib import Path
import re
import time

EPSILON = 1e-9
REPEATS = 1000


def validate_matrix(matrix, size, name="배열"):
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("{}: {}개 행이 필요합니다.".format(name, size))
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{}: 각 행에 {}개 열이 필요합니다.".format(name, size))
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{}: 숫자만 사용할 수 있습니다.".format(name))
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("{}: 유한한 숫자가 필요합니다.".format(name))
    return matrix


def mac_score(pattern, filter_matrix):
    """검증된 같은 크기의 N×N 배열을 받아 MAC 점수를 계산한다."""
    score = 0.0
    for row in range(len(pattern)):
        for col in range(len(pattern)):
            score += pattern[row][col] * filter_matrix[row][col]
    if not math.isfinite(score):
        raise ValueError("MAC 연산 결과가 유한한 숫자 범위를 벗어났습니다.")
    return score


def decide(score_a, score_b, label_a="Cross", label_b="X"):
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def normalize_label(value):
    if not isinstance(value, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    try:
        return labels[value.strip().lower()]
    except KeyError:
        raise ValueError("지원하지 않는 라벨: {!r}".format(value)) from None


def generate_filters(size):
    """N×N Cross/X 생성. 짝수 N은 인덱스 N//2의 행/열을 중심으로 쓴다."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("크기 N은 1 이상의 정수여야 합니다.")
    cross = [[int(r == size // 2 or c == size // 2)
              for c in range(size)] for r in range(size)]
    x = [[int(r == c or r + c == size - 1)
          for c in range(size)] for r in range(size)]
    return {"Cross": cross, "X": x}


def flatten(matrix):
    """행 우선 순서로 2차원 배열을 길이 N²의 1차원 배열로 바꾼다."""
    return [value for row in matrix for value in row]


def mac_score_flat(pattern, filter_values):
    """검증된 같은 길이의 1차원 배열을 순서대로 곱하고 누적한다."""
    score = 0.0
    for index in range(len(pattern)):
        score += pattern[index] * filter_values[index]
    if not math.isfinite(score):
        raise ValueError("MAC 연산 결과가 유한한 숫자 범위를 벗어났습니다.")
    return score


def benchmark(pattern, filter_matrix, repeats, operation=mac_score):
    # 검증, JSON 읽기, 화면 출력은 측정 구간에서 제외한다.
    operation(pattern, filter_matrix)
    start = time.perf_counter()
    for _ in range(repeats):
        operation(pattern, filter_matrix)
    return (time.perf_counter() - start) * 1000 / repeats


def compare_layouts(pattern, filter_matrix, repeats):
    start = time.perf_counter()
    flat_pattern, flat_filter = flatten(pattern), flatten(filter_matrix)
    conversion_ms = (time.perf_counter() - start) * 1000
    before_score = mac_score(pattern, filter_matrix)
    after_score = mac_score_flat(flat_pattern, flat_filter)
    if before_score != after_score:
        raise ValueError("2차원/1차원 MAC 점수가 일치하지 않습니다.")
    before = benchmark(pattern, filter_matrix, repeats)
    after = benchmark(flat_pattern, flat_filter, repeats, mac_score_flat)
    return {"before_ms": before, "after_ms": after,
            "conversion_ms": conversion_ms, "score": before_score,
            "speedup": before / after if after else None}


def print_layout_comparison(repeats, cases):
    print("\n[보너스: 동일 입력 / 각각 {}회 / 평균 ms]".format(repeats))
    print("크기       2D MAC      1D MAC      변환 1회(ms)   속도비(2D/1D)")
    for size, pattern, filter_matrix in cases:
        result = compare_layouts(pattern, filter_matrix, repeats)
        ratio = "{:.2f}x".format(result["speedup"]) if result["speedup"] is not None else "측정 불가"
        print("{0}x{0:<5} {1:.6f}    {2:.6f}    {3:.6f}       {4}".format(
            size, result["before_ms"], result["after_ms"], result["conversion_ms"], ratio))
    print("점수 일치 확인 완료. 1D MAC 시간에는 최초 배열 변환 시간이 포함되지 않습니다.")
    print("배열 변환은 O(N²), MAC은 두 방식 모두 O(N²)입니다. 속도비는 실행 환경에 따라 달라집니다.")


def print_performance(repeats, cases):
    print("\n[성능 분석: 필터 1개 MAC 호출 평균 / {}회]".format(repeats))
    print("크기       평균 시간(ms)    MAC 횟수(N²)")
    for size, pattern, filter_matrix in cases:
        elapsed = benchmark(pattern, filter_matrix, repeats)
        print("{0}x{0:<6} {1:>12.6f}    {2}".format(size, elapsed, size * size))


def read_matrix(name, size=3):
    print("\n{}: 각 줄에 숫자 {}개를 공백으로 구분해 {}줄 입력하세요.".format(name, size, size))
    matrix = []
    while len(matrix) < size:
        try:
            row = [float(value) for value in input("{}행 > ".format(len(matrix) + 1)).split()]
            if len(row) != size or any(not math.isfinite(value) for value in row):
                raise ValueError
        except ValueError:
            print("입력 형식 오류: 유한한 숫자 {}개를 공백으로 구분해 입력하세요.".format(size))
            continue
        matrix.append(row)
    return matrix


def run_manual(repeats, generated_filters=False):
    if generated_filters:
        pair = generate_filters(3)
        filter_a, filter_b = pair["Cross"], pair["X"]
        print("생성한 3x3 필터 사용: A=Cross, B=X")
    else:
        filter_a = read_matrix("필터 A")
        filter_b = read_matrix("필터 B")
    print("필터 A, B 저장 완료")
    pattern = read_matrix("입력 패턴")
    score_a = mac_score(pattern, filter_a)
    score_b = mac_score(pattern, filter_b)
    print("\nA 점수: {:.16g}\nB 점수: {:.16g}".format(score_a, score_b))
    print("판정:", decide(score_a, score_b, "A", "B"))
    print_performance(repeats, [(3, pattern, filter_a)])
    print_layout_comparison(repeats, [(3, pattern, filter_a)])


def run_generator(repeats, size=None, pattern_label=None):
    while size is None:
        try:
            size = int(input("생성할 크기 N (1 이상) > "))
            if size < 1:
                raise ValueError
        except ValueError:
            print("크기 N은 1 이상의 정수여야 합니다.")
            size = None
    pair = generate_filters(size)
    for label, matrix in pair.items():
        print("\n{} ({}x{})".format(label, size, size))
        for row in matrix:
            print(" ".join(str(value) for value in row))
    while pattern_label not in ("cross", "x", "manual"):
        pattern_label = input("입력 패턴 선택 (cross / x / manual: 직접 입력) > ").strip().lower()
    pattern = (read_matrix("입력 패턴", size) if pattern_label == "manual"
               else pair[normalize_label(pattern_label)])
    cross_score = mac_score(pattern, pair["Cross"])
    x_score = mac_score(pattern, pair["X"])
    print("\nCross 점수: {:.16g}\nX 점수: {:.16g}".format(cross_score, x_score))
    print("판정:", decide(cross_score, x_score))
    cases = [(size, pattern, pair["Cross"])]
    print_performance(repeats, cases)
    print_layout_comparison(repeats, cases)


def load_filters(raw, size):
    if not isinstance(raw, dict):
        raise ValueError("size_{} 필터 객체가 없습니다.".format(size))
    result = {}
    for key, matrix in raw.items():
        label = normalize_label(key)
        if label in result:
            raise ValueError("중복 필터 라벨: {}".format(label))
        result[label] = validate_matrix(matrix, size, label)
    if set(result) != {"Cross", "X"}:
        raise ValueError("Cross와 X 필터가 모두 필요합니다.")
    return result


def analyze_data(data):
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위는 객체여야 합니다.")
    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        raise ValueError("filters와 patterns는 객체여야 합니다.")
    loaded, errors = {}, {}
    for size in (5, 13, 25):
        try:
            loaded[size] = load_filters(filters.get("size_{}".format(size)), size)
            print("size_{} 필터 로드 완료 (Cross, X)".format(size))
        except ValueError as exc:
            errors[size] = str(exc)
            print("size_{} 필터 오류: {}".format(size, exc))

    results, performance_inputs = [], {}
    for key, case in patterns.items():
        result = {"id": key, "passed": False}
        try:
            match = re.fullmatch(r"size_(\d+)_(\d+)", key)
            if not match:
                raise ValueError("패턴 키는 size_{N}_{idx} 형식이어야 합니다.")
            size = int(match.group(1))
            if size not in (5, 13, 25):
                raise ValueError("지원하지 않는 크기: {}".format(size))
            if size in errors:
                raise ValueError(errors[size])
            if not isinstance(case, dict):
                raise ValueError("패턴 항목은 객체여야 합니다.")
            pattern = validate_matrix(case.get("input"), size, "input")
            expected = normalize_label(case.get("expected"))
            pair = loaded[size]
            cross_score = mac_score(pattern, pair["Cross"])
            x_score = mac_score(pattern, pair["X"])
            prediction = decide(cross_score, x_score)
            result.update(prediction=prediction, expected=expected,
                          cross_score=cross_score, x_score=x_score,
                          passed=prediction == expected)
            if not result["passed"]:
                result["reason"] = ("동점: |Cross-X| < 1e-9" if prediction == "UNDECIDED"
                                    else "판정 {} / 정답 {} 불일치".format(prediction, expected))
            performance_inputs.setdefault(size, (pattern, pair["Cross"]))
            print("\n{}: Cross={:.16g}, X={:.16g}".format(key, cross_score, x_score))
            print("판정: {} | expected: {} | {}".format(
                prediction, expected, "PASS" if result["passed"] else "FAIL"))
        except (ValueError, OverflowError) as exc:
            result["reason"] = str(exc)
            print("\n{}: FAIL ({})".format(key, exc))
        results.append(result)
    return results, performance_inputs


def run_json(path, repeats):
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    results, measured = analyze_data(data)
    cases = []
    for size in (3, 5, 13, 25):
        if size in measured:
            pattern, filter_matrix = measured[size]
        else:
            pair = generate_filters(size)
            pattern = filter_matrix = pair["Cross"]
            print("성능 측정 {}x{}: 생성한 기준 배열 사용".format(size, size))
        cases.append((size, pattern, filter_matrix))
    print_performance(repeats, cases)
    print_layout_comparison(repeats, cases)
    passed = sum(result["passed"] for result in results)
    print("\n[결과 요약]\n총 테스트: {} | 통과: {} | 실패: {}".format(
        len(results), passed, len(results) - passed))
    for result in results:
        if not result["passed"]:
            print("- {}: {}".format(result["id"], result["reason"]))
    if not results:
        print("분석할 패턴이 없습니다.")
    return 0 if results and passed == len(results) else 1


def main():
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--mode", choices=("1", "2", "3"), help="1: 사용자 입력, 2: JSON 분석, 3: 패턴 생성기")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.json"))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--generated-filters", action="store_true", help="모드 1에서 생성한 3x3 필터 사용")
    parser.add_argument("--size", type=int, help="모드 3의 생성 크기 N")
    parser.add_argument("--pattern", choices=("cross", "x", "manual"), help="모드 3의 입력 패턴")
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("반복 횟수는 최소 10회여야 합니다.")
    if args.size is not None and args.size < 1:
        parser.error("크기 N은 1 이상의 정수여야 합니다.")
    print("=== Mini NPU Simulator ===")
    try:
        mode = args.mode
        while mode not in ("1", "2", "3"):
            print("1. 사용자 입력 (3x3)\n2. data.json 분석\n3. 보너스: N×N 패턴 생성기")
            mode = input("선택 > ").strip()
        if mode == "1":
            run_manual(args.repeats, args.generated_filters)
            return 0
        if mode == "3":
            run_generator(args.repeats, args.size, args.pattern)
            return 0
        return run_json(args.data, args.repeats)
    except (OSError, ValueError, OverflowError) as exc:
        print("오류:", exc)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

