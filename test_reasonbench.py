import unittest
from reasonbench import calculate, parse_direct, parse_tool, tasks, prompt, assess, summarize


class ReasonTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate('(19 + 5) * 3'), 72)

    def test_precedence(self):
        self.assertEqual(calculate('8+6*4'), 32)

    def test_negative(self):
        self.assertEqual(calculate('12-29'), -17)

    def test_fraction(self):
        self.assertEqual(calculate('(3/2)*4'), 6)
        with self.assertRaises(ValueError):
            calculate('1/2')

    def test_unsafe_syntax(self):
        for value in ('__import__("os").system("id")', '[1][0]', '2**99', 'True', 'x+1', '1//2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculate(value)

    def test_zero_division(self):
        with self.assertRaises(ValueError):
            calculate('4/0')

    def test_resource_limits(self):
        for value in ('1'*81, '10001', '10000*10000*10000'):
            with self.assertRaises(ValueError):
                calculate(value)

    def test_direct_parser(self):
        self.assertEqual(parse_direct('Answer: -17.'), -17)

    def test_verbose_rejected(self):
        with self.assertRaises(ValueError):
            parse_direct('43 because 17+26=43')

    def test_tool_parser(self):
        self.assertEqual(parse_tool('<calc>17+26</calc>'), 43)

    def test_multiple_calls_rejected(self):
        with self.assertRaises(ValueError):
            parse_tool('<calc>1</calc><calc>2</calc>')

    def test_wrong_expression_not_corrected(self):
        self.assertEqual(assess('<calc>1+1</calc>', 'calculator', 43)['status'], 'wrong_answer')

    def test_invalid_output_counted(self):
        self.assertEqual(assess('hello', 'calculator', 43)['status'], 'invalid_output')

    def test_task_ids_unique(self):
        self.assertEqual(len({task['id'] for task in tasks()}), 12)

    def test_prompt_has_no_gold(self):
        self.assertTrue(prompt('QUESTION', 'calculator')[-1]['content'].startswith('QUESTION\n'))

    def test_summary(self):
        row = {'mode': 'direct', 'correct': True, 'status': 'correct', 'generation_seconds': 1}
        self.assertEqual(summarize([row])['direct']['correct'], 1)


if __name__ == '__main__':
    unittest.main()
