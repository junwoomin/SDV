"""Actual committed-sample counters shared by collection events (no CARLA import)."""


def save_image(writer, path, image):
    """OpenCV can return False on write failure; never count that sample as saved."""
    if not writer(str(path), image):
        raise IOError("수집 이미지 저장 실패: {}. 저장 경로, 디스크 공간과 권한을 확인하세요.".format(path))


class CollectionProgress:
    def __init__(self, towns, per_town):
        self.towns = tuple(towns)
        self.per_town = int(per_town)
        if not self.towns or self.per_town < 1 or len(set(self.towns)) != len(self.towns):
            raise ValueError("수집 맵과 맵별 목표 샘플 수가 올바르지 않습니다.")
        self.counts = dict.fromkeys(self.towns, 0)

    def snapshot(self, town=None):
        saved = sum(self.counts.values())
        total = len(self.towns) * self.per_town
        event = {"saved": saved, "total": total, "progress": saved / total}
        if town is not None:
            event.update(town=town, town_saved=self.counts[town], town_target=self.per_town)
        return event

    def commit(self, town):
        """Called only after every requested file for the sample was written."""
        if self.counts[town] >= self.per_town:
            raise ValueError("맵의 목표 샘플 수를 초과했습니다: {}".format(town))
        self.counts[town] += 1
        return self.snapshot(town)
