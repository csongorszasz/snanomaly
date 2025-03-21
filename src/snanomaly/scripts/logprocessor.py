import json
from pathlib import Path

from attrs import define, field
from loguru import logger

from snanomaly import dirs


@define
class FilterCriteria:
    """
    Define filter criteria for log processing.
    """

    keyword: str = field()
    negate: bool = field(default=False)

    def matches(self, line: str) -> bool:
        if self.negate:
            return self.keyword not in line
        return self.keyword in line

@define
class LogProcessor:
    """
    Organize, format, filter logs written to files.

    The format of the log files is assumed to follow the `loguru` package's serialized log format.
    """

    log_target: Path = field(default=None) # Path to a log file or directory (in the case of multiple files)

    def __attrs_post_init__(self):
        if not self.log_target.exists():
            raise FileNotFoundError(f"Log target {self.log_target} does not exist.")
        if not self.log_target.is_dir() and not self.log_target.is_file():
            raise ValueError(f"Log target {self.log_target} is neither a file nor a directory.")

    def process_logs(self, filter_criteria: FilterCriteria = None) -> list[str]:
        """
        Process logs from the log target. If it's a directory, process all files in it.
        """
        kept_lines = []
        if self.log_target.is_dir():
            for file in self.log_target.iterdir():
                if file.is_file():
                    kept_lines.extend(self._process_log_file(file, filter_criteria))
        else:
            kept_lines.extend(self._process_log_file(self.log_target, filter_criteria))
        return kept_lines

    def _process_log_file(self, file: Path, filter_criteria: FilterCriteria = None) -> list[str]:
        """
        Process a single log file.
        """
        kept_lines = []
        with file.open() as f:
            for line in f:
                if filter_criteria is None or filter_criteria.matches(line):
                    kept_lines.append(line)
        return kept_lines

    @classmethod
    def lines_to_files(cls, lines: list[str], collection_name: str):
        """
        Write each given line to a separate file with json pretty print.
        Target directory:
            `logs/filtered/{collection_name}/`

        Also write the log message itself (text attribute) to a separate file with the same name but with `_text`
        suffix.
        """
        if not lines:
            logger.warning("No lines to write.")
            return
        collection_dir = dirs.LOGS_FILTERED / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(lines):
            line_dict = json.loads(line)
            with (collection_dir / f"{i+1}.json").open("w") as f:
                json.dump(line_dict, f, indent=4)
            with (collection_dir / f"{i+1}_text.txt").open("w") as f:
                text = line_dict.get("text")
                f.write(text)
        logger.info(f"See filtered logs at: `{collection_dir}`.")
