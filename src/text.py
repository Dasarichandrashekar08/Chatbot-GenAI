from flask import json
import yaml


def nonewlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")


def load_json_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def load_yml_content_from_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
