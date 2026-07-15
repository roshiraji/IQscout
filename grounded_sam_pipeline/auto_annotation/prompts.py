"""Text prompt enhancement for Grounding DINO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassPromptMap:
    """Maps dataset class names to enhanced prompts and back to class IDs."""

    classes: tuple[str, ...]
    prompts: tuple[str, ...]
    class_to_id: dict[str, int]
    prompt_to_class: dict[str, str]

    def id_for_class(self, class_name: str) -> int:
        return self.class_to_id[class_name]

    def class_for_prompt(self, prompt: str) -> str | None:
        return self.prompt_to_class.get(prompt)


def _default_prompt(class_name: str, template: str) -> str:
    return template.format(class_name=class_name)


def build_class_prompts(
    classes: list[str],
    template: str = "all {class_name}s",
    overrides: dict[str, str] | None = None,
) -> ClassPromptMap:
    overrides = overrides or {}
    prompts: list[str] = []
    class_to_id: dict[str, int] = {}
    prompt_to_class: dict[str, str] = {}

    for idx, cls in enumerate(classes):
        prompt = overrides.get(cls, _default_prompt(cls, template))
        prompts.append(prompt)
        class_to_id[cls] = idx
        prompt_to_class[prompt] = cls

    return ClassPromptMap(
        classes=tuple(classes),
        prompts=tuple(prompts),
        class_to_id=class_to_id,
        prompt_to_class=prompt_to_class,
    )


def match_phrase_to_class(phrase: str, prompt_map: ClassPromptMap) -> str | None:
    """Match Grounding DINO phrase output back to a dataset class."""
    phrase = phrase.strip().lower()
    for prompt, cls in prompt_map.prompt_to_class.items():
        if phrase == prompt.lower():
            return cls
    # Fuzzy: phrase contained in prompt or vice versa
    for prompt, cls in prompt_map.prompt_to_class.items():
        pl, cl = prompt.lower(), cls.lower()
        if phrase in pl or pl in phrase or phrase in cl or cl in phrase:
            return cls
    return None
