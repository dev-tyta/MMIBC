"""The backbone comparison without a Hugging Face token: gated rows are skipped cleanly, everything else runs, and adding the token later
trains only what is missing. (The gated DINOv3 weights themselves cannot be downloaded here; the machinery around them is what is tested.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from mmibc.models import backbones as bb
from mmibc.utils import hf
from mmibc.utils.config import load_config

TINY = ["model.shared_backbone=true", "model.pretrained=false", "model.freeze=none", "model.xattn_heads=2", "model.head_hidden=32", "training.epochs=1",
        "eval.controls=[none]", "eval.bootstrap_samples=20", "data.store_long_side=112", "training.batch_size=8", "input.image_size=56"]

SUITE = """config: vindr_multiview.yaml
reference: dinov2_tiny
continue_on_error: true
runs:
  - {name: dinov2_tiny,    set: [model.backbone=dinov2_tiny_test]}
  - {name: dinov2reg_tiny, set: [model.backbone=dinov2reg_tiny_test]}
  - {name: dinov3_tiny,    set: [model.backbone=dinov3_tiny_test, input.image_size=64]}
"""


@pytest.fixture()
def no_token(monkeypatch):
    for k in hf.TOKEN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hf, "hf_token", lambda: None)
    monkeypatch.setattr(hf, "is_cached", lambda *a, **k: False)


def test_gated_backbone_without_token_is_unavailable_with_a_fix(no_token):
    reason = hf.unavailable_reason("dinov3_vitb16")
    assert reason and "facebook/dinov3-vitb16-pretrain-lvd1689m" in reason and "HF_TOKEN" in reason and "export_colab_secrets" in reason
    for name in ("dinov2_base", "dinov2reg_base", "tv_convnext_base", "dinov3_tiny_test", "no_such_backbone"):
        assert hf.unavailable_reason(name) is None                                  # ungated, tiny, or not ours to judge


def test_token_or_a_local_copy_makes_a_gated_backbone_available(no_token, monkeypatch):
    monkeypatch.setattr(hf, "is_cached", lambda repo, *a, **k: repo == "facebook/dinov3-vitl16-pretrain-lvd1689m")
    assert hf.unavailable_reason("dinov3_vitl16") is None and hf.unavailable_reason("dinov3_vitb16")   # a model downloaded earlier needs no token
    monkeypatch.setattr(hf, "hf_token", lambda: "hf_dummy")
    assert hf.unavailable_reason("dinov3_vitb16") is None


def test_the_token_is_found_in_either_environment_variable_and_exported(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_other")
    assert hf.hf_token() == "hf_other" and hf.ensure_hf_token() == "hf_other"
    import os

    assert os.environ["HF_TOKEN"] == "hf_other"                                    # from_pretrained reads HF_TOKEN
    monkeypatch.setenv("HF_TOKEN", "hf_first")
    assert hf.hf_token() == "hf_first"


def test_a_refused_gated_download_says_what_to_do(monkeypatch):
    class Boom(Exception):
        pass

    def refuse(*a, **k):
        raise Boom("401 Client Error: gated repo")

    import transformers

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", refuse)
    with pytest.raises(RuntimeError) as e:
        bb.ViTBackbone("dinov3_vitb16", pretrained=True)
    msg = str(e.value)
    assert "GATED" in msg and "HF_TOKEN" in msg and "not confirmed from its model card" in msg      # B/16 id is the unconfirmed one
    with pytest.raises(RuntimeError) as e2:
        bb.ViTBackbone("dinov3_vitl16", pretrained=True)
    assert "GATED" in str(e2.value) and "not confirmed" not in str(e2.value)


def test_suite_runs_what_it_can_skips_what_it_cannot_and_a_rerun_trains_only_the_rest(project, tmp_path, monkeypatch):
    from mmibc.suite import run_suite

    paths, over = project
    suite = tmp_path / "bb_mini.yaml"
    suite.write_text(SUITE)
    blocked = {"dinov3_tiny_test"}
    monkeypatch.setattr("mmibc.suite.unavailable_reason", lambda name: f"{name} is gated; no token" if name in blocked else None)   # stands in for a gated row

    res = run_suite(suite, paths, [*over, *TINY])
    out = paths.runs / "suite_bb_mini"
    st = pd.read_csv(out / "status.csv").set_index("run")["state"]
    assert st["dinov2_tiny"] == "done" and st["dinov2reg_tiny"] == "done" and st["dinov3_tiny"].startswith("SKIPPED")
    assert set(res["model"]) == {"dinov2_tiny", "dinov2reg_tiny", "dinov3_tiny"}
    skipped = res[res["model"] == "dinov3_tiny"].iloc[0]
    assert str(skipped["error"]).startswith("SKIPPED") and "no token" in skipped["error"]
    done = res[res["model"].isin(["dinov2_tiny", "dinov2reg_tiny"])]
    assert done["auroc"].notna().all()                                              # the comparison table exists with the rows that could run
    assert not list(out.glob("FAILED_*.txt")) and not (paths.runs / "bb_mini_dinov3_tiny_dinov3_tiny_test").exists()   # skipped before anything was built

    stamps = {n: (paths.runs / f"bb_mini_{n}_{b}" / "metrics.json").stat().st_mtime_ns for n, b in (("dinov2_tiny", "dinov2_tiny_test"), ("dinov2reg_tiny", "dinov2reg_tiny_test"))}
    blocked.clear()                                                                 # the token has been added
    res2 = run_suite(suite, paths, [*over, *TINY])
    st2 = pd.read_csv(out / "status.csv").set_index("run")["state"]
    assert (st2 == "done").all() and len(st2) == 3
    assert set(res2["model"]) == {"dinov2_tiny", "dinov2reg_tiny", "dinov3_tiny"} and ("error" not in res2.columns or res2["error"].isna().all())
    for n, b in (("dinov2_tiny", "dinov2_tiny_test"), ("dinov2reg_tiny", "dinov2reg_tiny_test")):
        assert (paths.runs / f"bb_mini_{n}_{b}" / "metrics.json").stat().st_mtime_ns == stamps[n]      # finished rows were not trained again
    assert "auroc" in res2.columns and res2["auroc"].notna().all()


def test_a_failing_row_does_not_stop_the_suite_and_is_recorded(project, tmp_path):
    from mmibc.suite import run_suite

    paths, over = project
    suite = tmp_path / "bb_fail.yaml"
    suite.write_text("config: vindr_multiview.yaml\ncontinue_on_error: true\nreference: dinov2_tiny\nruns:\n"
                     "  - {name: dinov2_tiny, set: [model.backbone=dinov2_tiny_test]}\n"
                     "  - {name: broken,      set: [model.backbone=dinov2_tiny_test, input.image_size=57]}\n")   # 57 is not a multiple of the patch size 14
    res = run_suite(suite, paths, [*over, *TINY])
    st = pd.read_csv(paths.runs / "suite_bb_fail" / "status.csv").set_index("run")["state"]
    assert st["dinov2_tiny"] == "done" and st["broken"].startswith("FAILED")
    assert (paths.runs / "suite_bb_fail" / "FAILED_broken.txt").exists()
    assert "error" in res.columns and res[res["model"] == "broken"]["error"].iloc[0].startswith("ValueError")


def test_backbone_suites_only_name_backbones_that_exist():
    from mmibc.suite import load_suite

    for suite in ("backbones", "backbones_cmmd"):
        spec = load_suite(suite)
        names = [r["name"] for r in spec["runs"]]
        assert spec["continue_on_error"] and spec["reference"] in names
        for r in spec["runs"]:
            cfg = load_config(spec["config"], r["set"])
            assert cfg.model.backbone in bb.BACKBONES, r
        gated = [r["name"] for r in spec["runs"] if bb.BACKBONES[load_config(spec["config"], r["set"]).model.backbone].gated]
        assert gated == ["dinov3_vitb16", "dinov3_vitl16"]                          # exactly the rows that need the token


# ----------------------------------------------------------------------------------------------- a damaged cached download
def _damaged_cache(tmp_path, monkeypatch):
    """A fake Hugging Face cache holding a broken dinov2-base next to an unrelated model; the real caches are never touched."""
    hub = tmp_path / "hf" / "hub"
    entry = hub / "models--facebook--dinov2-base"
    (entry / "blobs").mkdir(parents=True)
    (entry / "blobs" / "model").write_bytes(b"\xff" * 64)
    other = hub / "models--facebook--dinov2-small"
    other.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(hf, "hub_cache_dirs", lambda: [hub])
    return hub, entry, other


def test_a_damaged_cached_download_is_removed_and_fetched_again_once(tmp_path, monkeypatch):
    import transformers

    hub, entry, other = _damaged_cache(tmp_path, monkeypatch)
    calls = []

    def flaky(repo, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("Error while deserializing header: header too large")
        assert not entry.exists()                                                 # removed BEFORE the second download
        return transformers.AutoModel.from_config(bb._tiny_config(bb.BACKBONES["dinov2_base"]))

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", flaky)
    m = bb.ViTBackbone("dinov2_base", pretrained=True)
    assert m.embed_dim > 0 and len(calls) == 2
    assert "force_download" not in calls[0] and calls[1]["force_download"] is True
    assert calls[0]["cache_dir"] == str(hub)                                      # the Drive cache, whatever the import order was
    assert not entry.exists() and other.exists()                                  # only the damaged repository went


def test_other_load_errors_leave_the_cache_alone(tmp_path, monkeypatch):
    import transformers

    hub, entry, other = _damaged_cache(tmp_path, monkeypatch)
    calls = []

    def refuse(repo, **kw):
        calls.append(kw)
        raise OSError("401 Client Error: gated repo, access denied")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", refuse)
    with pytest.raises(RuntimeError, match="Could not load backbone"):
        bb.ViTBackbone("dinov2_base", pretrained=True)
    assert len(calls) == 1 and entry.exists()                                     # not a damaged file: nothing deleted, nothing retried


def test_a_download_that_is_still_unreadable_says_what_to_delete(tmp_path, monkeypatch):
    import transformers

    hub, entry, other = _damaged_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda repo, **kw: (_ for _ in ()).throw(RuntimeError("Error while deserializing header: header too large")))
    with pytest.raises(RuntimeError) as e:
        bb.ViTBackbone("dinov2_base", pretrained=True)
    assert "models--facebook--dinov2-base" in str(e.value) and "by hand" in str(e.value)


def test_damage_is_recognised_through_a_chain_and_only_the_named_repository_is_purged(tmp_path, monkeypatch):
    try:
        try:
            raise ValueError("Error while deserializing header: MetadataIncompleteBuffer")
        except ValueError as inner:
            raise RuntimeError("could not load") from inner
    except RuntimeError as outer:
        assert hf.looks_corrupt(outer)
    assert not hf.looks_corrupt(OSError("401 Client Error")) and not hf.looks_corrupt(ConnectionError("Network is unreachable"))
    hub, entry, other = _damaged_cache(tmp_path, monkeypatch)
    assert hf.purge_cached_repo("facebook/dinov2-base") == [entry] and other.exists()
    assert hf.purge_cached_repo("facebook/dinov2-base") == []                      # already gone: nothing to do, no error


def test_the_cache_directory_follows_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "x"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert hf.hf_cache_kwargs() == {"cache_dir": str(tmp_path / "x" / "hub")} and tmp_path / "x" / "hub" in hf.hub_cache_dirs()
    monkeypatch.delenv("HF_HOME")
    assert hf.hf_cache_kwargs() == {}


def test_a_damaged_cnn_checkpoint_is_removed_and_fetched_again(tmp_path, monkeypatch):
    import torchvision.models as tvm

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    url = tvm.get_model_weights("convnext_base").DEFAULT.url
    damaged = tmp_path / "torch" / "hub" / "checkpoints" / url.rsplit("/", 1)[-1]
    damaged.parent.mkdir(parents=True)
    damaged.write_bytes(b"not a zip")
    calls = []
    real = tvm.convnext_base

    def flaky(weights=None, **kw):
        calls.append(weights)
        if len(calls) == 1:
            raise RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory")
        assert not damaged.exists()
        return real(weights=None)

    monkeypatch.setattr(tvm, "convnext_base", flaky)
    m = bb.TVBackbone("tv_convnext_base", pretrained=True)
    assert m.embed_dim > 0 and len(calls) == 2 and not damaged.exists()
