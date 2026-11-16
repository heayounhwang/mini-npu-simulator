"""실행: python -m unittest discover -s mission3 -v"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import main as app


def sample_data():
    data = {"filters": {}, "patterns": {}}
    for size in (5, 13, 25):
        pair = app.generate_filters(size)
        data["filters"]["size_{}".format(size)] = {
            "cross": pair["Cross"], "x": pair["X"]}
        for index, (label, expected) in enumerate((("Cross", "+"), ("X", "x")), 1):
            data["patterns"]["size_{}_{}".format(size, index)] = {
                "input": pair[label], "expected": expected}
    return data


class SimulatorTests(unittest.TestCase):
    def test_manual_tie_cli_uses_korean_display(self):
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(Path(app.__file__)), "--mode", "1",
             "--generated-filters", "--repeats", "10"],
            input="0 0 0\n0 0 0\n0 0 0\n", capture_output=True,
            encoding="utf-8", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("판정: 판정 불가", result.stdout)
        self.assertNotIn("판정: UNDECIDED", result.stdout)

    def test_label_variants_and_rejection(self):
        for value, expected in (("+", "Cross"), ("cross", "Cross"),
                                (" Cross ", "Cross"), ("x", "X"), (" X ", "X")):
            self.assertEqual(app.normalize_label(value), expected)
        for invalid in ("?", "UNDECIDED", None):
            with self.assertRaises(ValueError):
                app.normalize_label(invalid)

    def test_generator_shapes_and_even_policy(self):
        for size in (1, 2, 3, 5, 8, 13, 25):
            pair = app.generate_filters(size)
            for label, matrix in pair.items():
                app.validate_matrix(matrix, size)
                self.assertTrue(all(v in (0, 1) for row in matrix for v in row))
            self.assertEqual(sum(map(sum, pair["Cross"])), 2 * size - 1)
            self.assertEqual(sum(map(sum, pair["X"])), 2 * size - size % 2)
            self.assertTrue(all(pair["Cross"][size // 2]))
            self.assertTrue(all(row[size // 2] for row in pair["Cross"]))
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                app.generate_filters(invalid)

    def test_flatten_order_and_score_equivalence(self):
        self.assertEqual(app.flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
        data = json.loads(Path(app.__file__).with_name("data.json").read_text(encoding="utf-8-sig"))
        for key, case in data["patterns"].items():
            size = int(key.split("_")[1])
            scores_2d, scores_1d = [], []
            for matrix in data["filters"]["size_{}".format(size)].values():
                scores_2d.append(app.mac_score(case["input"], matrix))
                scores_1d.append(app.mac_score_flat(app.flatten(case["input"]), app.flatten(matrix)))
            self.assertEqual(scores_2d, scores_1d)
            self.assertEqual(app.decide(*scores_2d), app.decide(*scores_1d))

    def test_layout_comparison_same_repetitions(self):
        pair = app.generate_filters(5)
        with patch.object(app, "benchmark", side_effect=[2.0, 1.0]) as timed:
            result = app.compare_layouts(pair["X"], pair["Cross"], 17)
        self.assertEqual([call.args[2] for call in timed.call_args_list], [17, 17])
        self.assertEqual(result["speedup"], 2.0)
        self.assertGreaterEqual(result["conversion_ms"], 0)

    def test_generator_cli_and_manual_filter_reuse(self):
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        for arguments, entries, expected in (
            (["--mode", "3", "--size", "7", "--pattern", "x"], "", "판정: X"),
            (["--mode", "3", "--size", "4", "--pattern", "cross"], "", "판정: Cross"),
            (["--mode", "1", "--generated-filters"], "1 0 1\n0 1 0\n1 0 1\n", "판정: B"),
            (["--mode", "3", "--size", "3", "--pattern", "manual"],
             "0 1 0\n1 1 1\n0 1 0\n", "판정: Cross"),
            (["--mode", "3"], "bad\n0\n3\nx\n", "판정: X"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, str(Path(app.__file__))] + arguments + ["--repeats", "10"],
                    input=entries, capture_output=True, encoding="utf-8", env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
                self.assertIn("점수 일치 확인 완료", result.stdout)

    def test_provided_data_regression(self):
        path = Path(app.__file__).with_name("data.json")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        results = self.analyze(data)
        self.assertEqual(len(results), 6)
        self.assertEqual([r["id"] for r in results if not r["passed"]],
                         ["size_5_1", "size_13_2", "size_25_1"])
        self.assertTrue(all(r["prediction"] == "UNDECIDED"
                            for r in results if not r["passed"]))

    def test_json_cli_reports_all_cases(self):
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(Path(app.__file__)), "--mode", "2", "--repeats", "10"],
            capture_output=True, encoding="utf-8", env=env)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("총 테스트: 6 | 통과: 3 | 실패: 3", result.stdout)
        for size in (3, 5, 13, 25):
            self.assertIn("{}x{}".format(size, size), result.stdout)

    def analyze(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return app.analyze_data(data)[0]

    def test_known_scores(self):
        pair = app.generate_filters(3)
        self.assertEqual(app.mac_score(pair["Cross"], pair["Cross"]), 5.0)
        self.assertEqual(app.mac_score(pair["Cross"], pair["X"]), 1.0)
        self.assertEqual(app.mac_score(pair["X"], pair["X"]), 5.0)

    def test_epsilon_boundary(self):
        self.assertEqual(app.decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(app.decide(0, app.EPSILON * 0.99), "UNDECIDED")
        self.assertEqual(app.decide(app.EPSILON, 0), "Cross")
        self.assertEqual(app.decide(0, app.EPSILON), "X")

    def test_all_sizes_and_labels(self):
        results = self.analyze(sample_data())
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r["passed"] for r in results))

    def test_bad_case_does_not_stop_next_case(self):
        mutations = [
            {"input": [[0]], "expected": "+"},
            {"input": [["bad"] * 5 for _ in range(5)], "expected": "+"},
            {"input": [[True] * 5 for _ in range(5)], "expected": "+"},
            {"input": [[float("nan")] * 5 for _ in range(5)], "expected": "+"},
            {"input": [[10 ** 500] * 5 for _ in range(5)], "expected": "+"},
            {"input": [[1e308] * 5 for _ in range(5)], "expected": "+"},
            {"input": [[0] * 5 for _ in range(5)], "expected": "?"},
            None,
        ]
        for case in mutations:
            with self.subTest(case_type=type(case).__name__):
                data = sample_data()
                data["patterns"]["size_5_1"] = case
                results = self.analyze(data)
                self.assertFalse(results[0]["passed"])
                self.assertTrue(results[1]["passed"])
                self.assertEqual(len(results), 6)

    def test_tie_and_wrong_expected_are_failures(self):
        data = sample_data()
        data["patterns"]["size_5_1"]["input"] = [[0] * 5 for _ in range(5)]
        data["patterns"]["size_5_2"]["expected"] = "+"
        results = self.analyze(data)
        self.assertEqual(results[0]["prediction"], "UNDECIDED")
        self.assertFalse(results[0]["passed"])
        self.assertFalse(results[1]["passed"])

    def test_filter_error_is_local_to_size(self):
        data = sample_data()
        data["filters"]["size_5"]["cross"] = [[0]]
        results = self.analyze(data)
        self.assertEqual(sum(r["passed"] for r in results), 4)

    def test_bad_key_and_missing_schema(self):
        data = sample_data()
        data["patterns"]["bad"] = {}
        results = self.analyze(data)
        self.assertEqual(len(results), 7)
        self.assertFalse(results[-1]["passed"])
        with self.assertRaises(ValueError):
            self.analyze({"filters": []})

    def test_input_retries_invalid_row(self):
        entries = ["1 2", "a b c", "nan 0 1", "0 1 0", "1 1 1", "0 1 0"]
        with patch("builtins.input", side_effect=entries), contextlib.redirect_stdout(io.StringIO()) as out:
            matrix = app.read_matrix("test")
        self.assertEqual(matrix, app.generate_filters(3)["Cross"])
        self.assertEqual(out.getvalue().count("입력 형식 오류"), 3)

    def test_manual_cli(self):
        entries = "0 1 0\n1 1 1\n0 1 0\n1 0 1\n0 1 0\n1 0 1\n1 0 1\n0 1 0\n1 0 1\n"
        import os
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(Path(app.__file__)), "--mode", "1", "--repeats", "10"],
            input=entries, capture_output=True, encoding="utf-8", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("A 점수: 1", result.stdout)
        self.assertIn("B 점수: 5", result.stdout)
        self.assertIn("판정: B", result.stdout)


if __name__ == "__main__":
    unittest.main()

