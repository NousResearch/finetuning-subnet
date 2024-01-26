import base64
import datetime
import shutil
import time
import unittest
from model.data import ModelId
import model.storage.disk.utils as utils
import os


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.base_dir = "test-models"
        self.sep = os.path.sep

    def tearDown(self):
        shutil.rmtree(path=self.base_dir, ignore_errors=True)

    def test_get_local_miners_dir(self):
        miners_dir = utils.get_local_miners_dir(self.base_dir)

        expected_path = self.base_dir + self.sep + "models"
        self.assertEqual(miners_dir, expected_path)

    def test_get_local_miner_dir(self):
        hotkey = "test-hotkey"

        miner_dir = utils.get_local_miner_dir(self.base_dir, hotkey)

        expected_path = self.base_dir + self.sep + "models" + self.sep + hotkey
        self.assertEqual(miner_dir, expected_path)

    def test_get_local_model_dir(self):
        hotkey = "test-hotkey"
        namespace = "test-namespace"
        name = "test-name"
        commit = "test-commit"
        model_id = ModelId(
            namespace=namespace, name=name, hash="test-hash", commit=commit
        )

        model_dir = utils.get_local_model_dir(self.base_dir, hotkey, model_id)

        expected_path = (
            self.base_dir
            + self.sep
            + "models"
            + self.sep
            + hotkey
            + self.sep
            + "models--"
            + namespace
            + "--"
            + name
        )
        self.assertEqual(model_dir, expected_path)

    def test_get_local_model_snapshot_dir(self):
        hotkey = "test-hotkey"
        namespace = "test-namespace"
        name = "test-name"
        commit = "test-commit"
        model_id = ModelId(
            namespace=namespace, name=name, hash="test-hash", commit=commit
        )

        model_dir = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model_id)

        expected_path = (
            self.base_dir
            + self.sep
            + "models"
            + self.sep
            + hotkey
            + self.sep
            + "models--"
            + namespace
            + "--"
            + name
            + self.sep
            + "snapshots"
            + self.sep
            + commit
        )
        self.assertEqual(model_dir, expected_path)

    def test_get_hf_download_path_dir(self):
        hotkey = "test-hotkey"
        namespace = "test-namespace"
        name = "test-name"
        commit = "test-commit"
        model_id = ModelId(
            namespace=namespace, name=name, hash="test-hash", commit=commit
        )

        hf_download_path_dir = utils.get_hf_download_path(
            utils.get_local_miner_dir(self.base_dir, hotkey), model_id
        )

        expected_path = (
            self.base_dir
            + self.sep
            + "models"
            + self.sep
            + hotkey
            + self.sep
            + "models--"
            + namespace
            + "--"
            + name
            + self.sep
            + "snapshots"
            + self.sep
            + commit
        )
        self.assertEqual(hf_download_path_dir, expected_path)

    def test_get_newest_datetime_under_path(self):
        file_name = "test.txt"
        path = self.base_dir + os.path.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        last_modified_expected = datetime.datetime.fromtimestamp(os.path.getmtime(path))

        last_modified_actual = utils.get_newest_datetime_under_path(self.base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_get_newest_datetime_under_path_empty(self):
        last_modified_expected = datetime.datetime.max

        last_modified_actual = utils.get_newest_datetime_under_path(self.base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_remove_dir_out_of_grace(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        # Sleep to ensure we are out of grace.
        time.sleep(1)

        self.assertTrue(os.path.exists(self.base_dir))
        deleted = utils.remove_dir_out_of_grace(self.base_dir, 0)
        self.assertTrue(deleted)
        self.assertFalse(os.path.exists(self.base_dir))

    def test_remove_dir_out_of_grace_in_grace(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        self.assertTrue(os.path.exists(self.base_dir))
        deleted = utils.remove_dir_out_of_grace(self.base_dir, 60)
        self.assertFalse(deleted)
        self.assertTrue(os.path.exists(self.base_dir))

    def test_get_hash_of_file(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        # Obtained by running openssl dgst -sha256 -binary test.txt | base64
        expected_file_hash = "tXNDvHVzqYIRiUx0rvK+M5+Lu4OLzhfPJH+gf7HvCeA="
        actual_file_hash = utils.get_hash_of_file(path)

        self.assertEqual(actual_file_hash, expected_file_hash)

    def test_get_hash_of_directory(self):
        # Make two sub directories.
        dir_1 = self.base_dir + self.sep + "dir1"
        dir_2 = self.base_dir + self.sep + "dir2"

        # Write the same two files to both sub directories.
        file_name_1 = "test1.txt"
        file_name_2 = "test2.txt"
        path_1_file_1 = dir_1 + os.path.sep + file_name_1
        path_1_file_2 = dir_1 + os.path.sep + file_name_2
        path_2_file_1 = dir_2 + os.path.sep + file_name_1
        path_2_file_2 = dir_2 + os.path.sep + file_name_2

        path_2_file_2 = dir_2 + os.path.sep + file_name_2
        file_paths = [path_1_file_1, path_1_file_2, path_2_file_1, path_2_file_2]

        os.mkdir(self.base_dir)
        os.mkdir(dir_1)
        os.mkdir(dir_2)

        for file_path in file_paths:
            file = open(file_path, "w")
            file.write("test text.")
            file.close()

        # Test that both sub directories have an equal hash.
        dir_1_hash = utils.get_hash_of_directory(dir_1)
        dir_2_hash = utils.get_hash_of_directory(dir_2)
        self.assertEqual(dir_1_hash, dir_2_hash)

        # Test that the hash for the overall directory does not equal the sub directory.
        base_dir_hash = utils.get_hash_of_directory(self.base_dir)
        self.assertNotEqual(base_dir_hash, dir_1_hash)

    def test_realize_symlinks_in_directory(self):
        end_file_dir = self.base_dir + self.sep + "end_files"
        symlink_source_dir = self.base_dir + self.sep + "symlink"

        regular_file = end_file_dir + self.sep + "test_file.txt"
        symlink_source = symlink_source_dir + self.sep + "symlink_source.txt"
        symlink_dest = end_file_dir + self.sep + "symlink_end.txt"

        # Make a regular file
        os.mkdir(self.base_dir)
        os.mkdir(end_file_dir)
        file = open(regular_file, "w")
        file.write("test text.")
        file.close()

        # Make a symlinked file
        os.mkdir(symlink_source_dir)
        file = open(symlink_source, "w")
        file.write("symlink source test text.")
        file.close()

        os.symlink(os.path.abspath(symlink_source), os.path.abspath(symlink_dest))

        # Confirm we see 3 files
        pre_file_count = 0
        for _, _, files in os.walk(self.base_dir):
            pre_file_count += len(files)
        self.assertEqual(pre_file_count, 3)

        realized_files = utils.realize_symlinks_in_directory(end_file_dir)

        # Confirm 1 file got realized and there are two total now.
        self.assertEqual(realized_files, 1)

        post_file_count = 0
        for _, _, files in os.walk(self.base_dir):
            post_file_count += len(files)
        self.assertEqual(post_file_count, 2)


if __name__ == "__main__":
    unittest.main()
