import json
import pathlib


def duration_to_seconds(hhmmss: str) -> int:
    split = hhmmss.split(":")
    assert len(split) <= 3
    split = (["0"] * (3 - len(split))) + split
    h, m, s = split
    return int(h) * 3600 + int(m) * 60 + int(s)
    

class Config:
    def __init__(self, config: dict[str, str | list[str] | int]) -> None:
        assert type(config["name"]) == str
        self.name: str = config["name"]
        assert type(config["path"]) == str
        self.path: str = config["path"]
        assert type(config["ad_songs"]) == list
        self.ad_songs: list[str] = config["ad_songs"]
        assert len(self.ad_songs) > 0

        if "ignore_first" in config:
            assert type(config["ignore_first"]) == str
            self.ignore_first_seconds: int = duration_to_seconds(config["ignore_first"])
        else:
            self.ignore_first_seconds: int = 0

        if "ignore_last" in config:
            assert type(config["ignore_last"]) == str
            self.ignore_last_seconds: int = duration_to_seconds(config["ignore_last"])
        else:
            self.ignore_last_seconds: int = 0

        assert type(config["minimum_ad_length"]) == str
        self.minimum_ad_length_seconds: int = duration_to_seconds(config["minimum_ad_length"])
        assert type(config["maximum_ad_length"]) == str
        self.maximum_ad_length_seconds = duration_to_seconds(config["maximum_ad_length"])
        assert type(config["ad_buffer"]) == str
        self.ad_buffer_seconds = duration_to_seconds(config["ad_buffer"])
        assert type(config["number_of_ad_breaks"]) == int
        self.number_of_ad_breaks: int = config["number_of_ad_breaks"]

def load_config(path: str | pathlib.Path) -> list[Config]:
    with open(path, "r") as fp:
        config = json.load(fp)
    assert type(config) == list
    return list(map(Config, config))
