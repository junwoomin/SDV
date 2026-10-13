"""Filesystem fixtures exercise successful saves and explicit write failures."""
import pytest

from sdv.collection.progress import CollectionProgress, save_image


def test_global_counter_does_not_reset_when_next_map_starts(tmp_path):
    progress = CollectionProgress(["Town01", "Town02"], 2)
    assert progress.snapshot() == {"saved": 0, "total": 4, "progress": 0}
    def fixture_writer(path, image):
        with open(path, "wb") as output:
            output.write(image)
        return True
    for town in ("Town01", "Town01", "Town02"):
        path = tmp_path / (town + "_" + str(progress.counts[town]) + ".png")
        save_image(fixture_writer, path, b"explicit encoded-image fixture")
        event = progress.commit(town)
    assert event == {"saved": 3, "total": 4, "progress": .75,
                     "town": "Town02", "town_saved": 1, "town_target": 2}
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert progress.snapshot()["saved"] == 3  # A stop cannot fill remaining samples.


def test_image_write_failure_does_not_count_the_uncommitted_sample(tmp_path):
    progress = CollectionProgress(["Town01"], 2)
    progress.commit("Town01")
    with pytest.raises(IOError, match="저장 실패"):
        save_image(lambda path, image: False, tmp_path / "missing.png", b"fixture")
        progress.commit("Town01")
    assert progress.snapshot() == {"saved": 1, "total": 2, "progress": .5}


def test_image_exception_preserves_original_disk_error(tmp_path):
    def full_disk(path, image):
        raise OSError("No space left on device")
    with pytest.raises(OSError, match="No space"):
        save_image(full_disk, tmp_path / "missing.png", b"fixture")


def test_completion_requires_all_commits():
    progress = CollectionProgress(["Town01", "Town02"], 1)
    assert progress.commit("Town01")["progress"] == .5
    assert progress.commit("Town02")["progress"] == 1
    with pytest.raises(ValueError, match="초과"):
        progress.commit("Town02")
