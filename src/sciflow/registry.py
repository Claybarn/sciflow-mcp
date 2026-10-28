"""Operation registry.

An operation declares the artifact kind of each input, the kind it outputs,
a pydantic model for its parameters and an integer version. The version is
part of every cache key and provenance record, so bump it whenever an
operation's numerical behaviour changes.

Input kinds are ``"series"``, ``"intervals"`` or ``"any"``. An output
kind of ``"same:<input>"`` means the output has the kind of that input.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from .types import Artifact


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoParams(Params):
    pass


@dataclass(frozen=True)
class OpSpec:
    name: str
    version: int
    category: str
    summary: str
    description: str
    inputs: dict[str, str]
    output: str
    params: type[Params]
    fn: Callable[..., Artifact]

    def output_kind(self, input_kinds: dict[str, str]) -> str:
        if self.output.startswith("same:"):
            return input_kinds.get(self.output.removeprefix("same:"), "any")
        return self.output

    def describe(self, full: bool = False) -> dict[str, Any]:
        d = {
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "summary": self.summary,
            "inputs": self.inputs,
            "output": self.output,
        }
        if full:
            d["description"] = self.description
            d["params_schema"] = self.params.model_json_schema()
        return d


REGISTRY: dict[str, OpSpec] = {}


def op(name: str, *, category: str, inputs: dict[str, str], output: str,
       params: type[Params] = NoParams, version: int = 1):
    def deco(fn):
        doc = inspect.cleandoc(fn.__doc__ or "")
        REGISTRY[name] = OpSpec(
            name=name, version=version, category=category,
            summary=doc.split("\n\n")[0].replace("\n", " "), description=doc,
            inputs=inputs, output=output, params=params, fn=fn,
        )
        return fn
    return deco


def get_op(name: str) -> OpSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown operation {name!r}; available: {sorted(REGISTRY)}") from None
