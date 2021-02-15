import unittest


class Tests(unittest.TestCase):
    def test_data_split(self):
        from data import train_test_split

        data = list(range(100))
        train_data, test_data = train_test_split(data, 0.9)

        self.assertEqual(len(train_data), 90)
        self.assertEqual(len(test_data), 10)
        self.assertEqual(len(set(train_data).intersection(set(test_data))), 0)


if __name__ == "__main__":
    unittest.main()
