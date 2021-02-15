import unittest


class Tests(unittest.TestCase):
    def test_data_split(self):
        """Tests train_test_split function"""
        from data import train_test_split

        data = list(range(100))
        train_data, test_data = train_test_split(data, 0.9)

        self.assertEqual(len(train_data), 90)
        self.assertEqual(len(test_data), 10)
        self.assertEqual(len(set(train_data).intersection(set(test_data))), 0)

    def test_polynomial_lang(self):
        """Tests PolynomialLang class."""
        from utils import load_file
        from data import PolynomialLanguage

        pairs = load_file("data/train.txt")
        lang = PolynomialLanguage()

        for src, trg in pairs:
            # test that sentence == sentence_to_words(words_to_sentence(sentence))
            src_reconstructed = lang.sentence_to_words(src)
            src_reconstructed = lang.words_to_sentence(src_reconstructed)
            self.assertEqual(src, src_reconstructed)

            trg_reconstructed = lang.sentence_to_words(trg)
            trg_reconstructed = lang.words_to_sentence(trg_reconstructed)
            self.assertEqual(trg, trg_reconstructed)


if __name__ == "__main__":
    unittest.main()
