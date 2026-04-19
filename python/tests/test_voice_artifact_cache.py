import numpy as np
import torch

import main as api_main


class FakeVoiceCloneModel:
    def __init__(self) -> None:
        self.prompt_calls = 0

    def _normalize_audio_inputs(
        self, ref_audio: tuple[np.ndarray, int] | list[tuple[np.ndarray, int]]
    ) -> list[tuple[np.ndarray, int]]:
        items = ref_audio if isinstance(ref_audio, list) else [ref_audio]
        normalized: list[tuple[np.ndarray, int]] = []
        for audio, sample_rate in items:
            normalized.append(
                (np.ascontiguousarray(audio.astype(np.float32)), int(sample_rate))
            )
        return normalized

    def create_voice_clone_prompt(
        self,
        ref_audio: tuple[np.ndarray, int],
        ref_text: str | None,
        x_vector_only_mode: bool,
    ) -> list[api_main.VoiceClonePromptItem]:
        assert x_vector_only_mode is False
        self.prompt_calls += 1
        wav, sample_rate = self._normalize_audio_inputs(ref_audio)[0]
        ref_code = torch.from_numpy((wav * 1000).round().astype(np.int64)).unsqueeze(1)
        ref_spk_embedding = torch.tensor(
            [float(wav.mean()), float(wav.std()), float(sample_rate)],
            dtype=torch.float32,
        )
        return [
            api_main.VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text=ref_text,
            )
        ]


def _sample_audio(seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(32, dtype=np.float32), 24000


def test_voice_artifact_cache_key_is_stable() -> None:
    ref_audio_a = _sample_audio(1)
    ref_audio_b = (ref_audio_a[0].copy(), ref_audio_a[1])

    key_a = api_main._voice_artifact_cache_key(
        ref_audio_a,
        ref_text="hello",
        use_icl=True,
    )
    key_b = api_main._voice_artifact_cache_key(
        ref_audio_b,
        ref_text="hello",
        use_icl=True,
    )

    assert key_a == key_b


def test_voice_artifact_cache_evicts_lru(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_TTS_VOICE_CACHE_SIZE", "2")
    api_main._clear_voice_artifact_cache()
    model = FakeVoiceCloneModel()

    first_audio = _sample_audio(1)
    second_audio = _sample_audio(2)
    third_audio = _sample_audio(3)

    first_key = api_main._voice_artifact_cache_key(
        first_audio,
        ref_text="a",
        use_icl=True,
    )
    second_key = api_main._voice_artifact_cache_key(
        second_audio,
        ref_text="b",
        use_icl=True,
    )
    third_key = api_main._voice_artifact_cache_key(
        third_audio,
        ref_text="c",
        use_icl=True,
    )

    api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=first_audio,
        ref_text="a",
        use_icl=True,
    )
    api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=second_audio,
        ref_text="b",
        use_icl=True,
    )
    api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=third_audio,
        ref_text="c",
        use_icl=True,
    )

    keys = api_main._voice_artifact_cache_keys_for_tests()
    assert first_key not in keys
    assert keys == [second_key, third_key]


def test_voice_artifact_cache_hit_bypasses_encoder_calls() -> None:
    api_main._clear_voice_artifact_cache()
    model = FakeVoiceCloneModel()
    ref_audio = _sample_audio(4)

    api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=ref_audio,
        ref_text="same text",
        use_icl=True,
    )
    cached_prompt = api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=ref_audio,
        ref_text="same text",
        use_icl=True,
    )

    assert model.prompt_calls == 1
    assert cached_prompt[0].icl_mode is True
    assert cached_prompt[0].x_vector_only_mode is False


def test_cached_voice_artifacts_match_uncached_path() -> None:
    api_main._clear_voice_artifact_cache()
    model = FakeVoiceCloneModel()
    ref_audio = _sample_audio(5)

    uncached = api_main._compute_voice_artifacts(model, ref_audio)
    cached_prompt = api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=ref_audio,
        ref_text="reference text",
        use_icl=True,
    )
    cache_hit_prompt = api_main._prepare_voice_clone_prompt(
        model,
        ref_audio=ref_audio,
        ref_text="reference text",
        use_icl=True,
    )

    assert torch.equal(cached_prompt[0].ref_code, uncached.ref_code)
    assert torch.equal(cache_hit_prompt[0].ref_code, uncached.ref_code)
    assert torch.equal(
        cached_prompt[0].ref_spk_embedding,
        uncached.ref_spk_embedding,
    )
    assert torch.equal(
        cache_hit_prompt[0].ref_spk_embedding,
        uncached.ref_spk_embedding,
    )
