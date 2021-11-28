from blather import Blather
import unittest

blather = Blather()

class TestBlather(unittest.TestCase):

    def test_read(self):
        self.assertEqual(type(blather.read(["Sample Text 1", "Sample Text 2", "Sample Text 3"])),type([]))

    def test_write(self):
        self.assertEqual(type(blather.write("sample_text")), type("Sample"))

if __name__ == '__main__':
    unittest.main()